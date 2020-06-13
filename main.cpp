#include "rt.hpp"

#define UTILS_TL_IMPL
#define UTILS_TL_TMP_SIZE 1 << 27
#include "utils.hpp"

#include <marl/scheduler.h>
#include <marl/thread.h>
#include <marl/waitgroup.h>

#define TRACY_ENABLE 1
#define TRACY_NO_EXIT 1
#include <tracy/Tracy.hpp>

#define PACK_SIZE 16

struct vCollision {
  vfloat3 pos;
  vfloat3 norm;
  vfloat  t;
};

struct vRay {
  vfloat3 o;
  vfloat3 d;
};

struct Collision {
  u32    mesh_id, face_id;
  float3 position;
  float3 normal;
  float  t, u, v;
};

// Möller–Trumbore intersection algorithm
static bool ray_triangle_test_moller(vec3 ray_origin, vec3 ray_dir, vec3 v0,
                                     vec3 v1, vec3 v2,
                                     Collision &out_collision) {
  float invlength = 1.0f / std::sqrt(glm::dot(ray_dir, ray_dir));
  ray_dir *= invlength;

  const float EPSILON = 1.0e-6f;
  vec3        edge1, edge2, h, s, q;
  float       a, f, u, v;
  edge1 = v1 - v0;
  edge2 = v2 - v0;
  h     = glm::cross(ray_dir, edge2);
  a     = glm::dot(edge1, h);
  if (a > -EPSILON && a < EPSILON)
    return false; // This ray is parallel to this triangle.
  f = 1.0 / a;
  s = ray_origin - v0;
  u = f * glm::dot(s, h);
  if (u < 0.0 || u > 1.0) return false;
  q = glm::cross(s, edge1);
  v = f * glm::dot(ray_dir, q);
  if (v < 0.0 || u + v > 1.0) return false;
  // At this stage we can compute t to find out where the intersection point
  // is on the line.
  float t = f * glm::dot(edge2, q);
  if (t > EPSILON) // ray intersection
  {
    out_collision.t      = t * invlength;
    out_collision.u      = u;
    out_collision.v      = v;
    out_collision.normal = glm::normalize(cross(edge1, edge2));
    out_collision.normal *= sign(-glm::dot(ray_dir, out_collision.normal));
    out_collision.position = ray_origin + ray_dir * t;

    return true;
  } else // This means that there is a line intersection but not a ray
         // intersection.
    return false;
}

// Woop intersection algorithm
static bool ray_triangle_test_woop(vec3 ray_origin, vec3 ray_dir, vec3 a,
                                   vec3 b, vec3 c, Collision &out_collision) {
  const float EPSILON        = 1.0e-4f;
  vec3        ab             = b - a;
  vec3        ac             = c - a;
  vec3        n              = cross(ab, ac);
  mat4        world_to_local = glm::inverse(mat4(
      //
      ab.x, ab.y, ab.z, 0.0f,
      //
      ac.x, ac.y, ac.z, 0.0f,
      //
      n.x, n.y, n.z, 0.0f,
      //
      a.x, a.y, a.z, 1.0f
      //
      ));
  vec4        ray_origin_local =
      (world_to_local * vec4(ray_origin.x, ray_origin.y, ray_origin.z, 1.0f));
  vec4 ray_dir_local =
      world_to_local * vec4(ray_dir.x, ray_dir.y, ray_dir.z, 0.0f);
  if (std::abs(ray_dir_local.z) < EPSILON) return false;
  float t = -ray_origin_local.z / ray_dir_local.z;
  if (t < EPSILON) return false;
  float u = ray_origin_local.x + t * ray_dir_local.x;
  float v = ray_origin_local.y + t * ray_dir_local.y;
  if (u > 0.0f && v > 0.0f && u + v < 1.0f) {
    out_collision.t        = t;
    out_collision.u        = u;
    out_collision.v        = v;
    out_collision.normal   = glm::normalize(n) * sign(-ray_dir_local.z);
    out_collision.position = ray_origin + ray_dir * t;
    return true;
  }
  return false;
}

struct Ray {
  float3 o;
  float3 d;
};

struct Tri {
  u32    id;
  float3 a;
  float3 b;
  float3 c;
  void   get_aabb(float3 &min, float3 &max) const {
    ito(3) min[i] = MIN(a[i], MIN(b[i], c[i]));
    ito(3) max[i] = MAX(a[i], MAX(b[i], c[i]));
  }
  float2 get_end_points(u8 dim, float3 min, float3 max) const {
    float3 sp;
    ito(i) sp[i] = MIN(a[i], MIN(b[i], c[i]));
    float3 ep;
    ito(i) ep[i] = MAX(a[i], MAX(b[i], c[i]));

    bool fully_inside = //
        sp.x > min.x && //
        sp.y > min.y && //
        sp.z > min.z && //
        ep.x < max.x && //
        ep.y < max.y && //
        ep.z < max.z && //
        true;
    if (fully_inside) return float2{sp[dim], ep[dim]};
  }
};

static_assert(sizeof(Tri) == 40, "Blamey!");

struct BVH_Node {
  // Bit layout:
  // +-------------------------+
  // | 32 31 30 29 28 27 26 25 |
  // | 24 23 22 21 20 19 18 17 |
  // | 16 15 14 13 12 11 10 9  |
  // | 8  7  6  5  4  3  2  1  |
  // +-------------------------+
  // +--------------+
  // | [32:32] Leaf |
  // +--------------+
  // |  Leaf:
  // +->+---------------------+---------------------+
  // |  | [31:25] Item count  | [24:1] Items offset |
  // |  +---------------------+---------------------+
  // |
  // |  Branch:
  // +->+----------------------------+
  //    | [24:1]  First child offset |
  //    +----------------------------+

  // constants
  static constexpr u32 LEAF_BIT = 1 << 31;
  // Leaf flags:
  static constexpr u32 ITEMS_OFFSET_MASK  = 0xffffff;  // 24 bits
  static constexpr u32 ITEMS_OFFSET_SHIFT = 0;         // low bits
  static constexpr u32 NUM_ITEMS_MASK     = 0b1111111; // 7 bits
  static constexpr u32 NUM_ITEMS_SHIFT    = 24;        // after first 24 bits
  static constexpr u32 MAX_ITEMS          = 2;         // max items
  // Node flags:
  static constexpr u32 FIRST_CHILD_MASK  = 0xffffff;
  static constexpr u32 FIRST_CHILD_SHIFT = 0;
  static constexpr u32 MAX_DEPTH         = 20;
  static constexpr f32 EPS               = 1.0e-3f;

  float3 min;
  float3 max;
  u32    flags;

  bool intersects(float3 tmin, float3 tmax) {
    return                 //
        tmax.x >= min.x && //
        tmin.x <= max.x && //
        tmax.y >= min.y && //
        tmin.y <= max.y && //
        tmax.z >= min.z && //
        tmin.z <= max.z && //
        true;
  }
  bool inside(float3 tmin) {
    return                 //
        tmin.x >= min.x && //
        tmin.x <= max.x && //
        tmin.y >= min.y && //
        tmin.y <= max.y && //
        tmin.z >= min.z && //
        tmin.z <= max.z && //
        true;
  }
  bool intersects_ray(float3 ro, float3 rd) {
    if (inside(ro)) return true;
    float3 invd = 1.0f / rd;
    float  dx_n = (min.x - ro.x) * invd.x;
    float  dy_n = (min.y - ro.y) * invd.y;
    float  dz_n = (min.z - ro.z) * invd.z;
    float  dx_f = (max.x - ro.x) * invd.x;
    float  dy_f = (max.y - ro.y) * invd.y;
    float  dz_f = (max.z - ro.z) * invd.z;
    float  nt   = MAX3(MIN(dx_n, dx_f), MIN(dy_n, dy_f), MIN(dz_n, dz_f));
    float  ft   = MIN3(MAX(dx_n, dx_f), MAX(dy_n, dy_f), MAX(dz_n, dz_f));
    return nt < ft + EPS;
  }
  void init_leaf(float3 min, float3 max, u32 offset) {
    flags = LEAF_BIT;
    ASSERT_DEBUG(offset <= ITEMS_OFFSET_MASK);
    flags |= ((offset << ITEMS_OFFSET_SHIFT));
    this->min = min;
    this->max = max;
  }
  void init_branch(float3 min, float3 max, BVH_Node *child) {
    ptrdiff_t diff = ((u8 *)child - (u8 *)this) / sizeof(BVH_Node);
    ASSERT_DEBUG(diff > 0 && diff < FIRST_CHILD_MASK);
    flags     = ((u32)diff << FIRST_CHILD_SHIFT);
    this->min = min;
    this->max = max;
  }
  bool is_leaf() { return (flags & LEAF_BIT) == LEAF_BIT; }
  u32  num_items() { return ((flags >> NUM_ITEMS_SHIFT) & NUM_ITEMS_MASK); }
  u32  items_offset() {
    return ((flags >> ITEMS_OFFSET_SHIFT) & ITEMS_OFFSET_MASK);
  }
  BVH_Node *first_child() {
    return this + (((flags >> FIRST_CHILD_SHIFT) & FIRST_CHILD_MASK));
  }
  void set_num_items(u32 num) {
    ASSERT_DEBUG(num <= NUM_ITEMS_MASK);
    flags &= ~(NUM_ITEMS_MASK << NUM_ITEMS_SHIFT);
    flags |= (num << NUM_ITEMS_SHIFT);
  }
  void add_item() { set_num_items(num_items() + 1); }
  bool is_full() { return num_items() == MAX_ITEMS - 1; }
};

struct BVH_Helper {
  float3      min;
  float3      max;
  Array<Tri>  tris;
  BVH_Helper *left;
  BVH_Helper *right;
  bool        is_leaf;
  void        init() {
    MEMZERO(*this);
    tris.init();
    min     = float3(1.0e10f, 1.0e10f, 1.0e10f);
    max     = float3(-1.0e10f, -1.0e10f, -1.0e10f);
    is_leaf = true;
  }
  void release() {
    if (left != NULL) left->release();
    if (right != NULL) right->release();
    tris.release();
    MEMZERO(*this);
    delete this;
  }
  void reserve(size_t size) { tris.reserve(size); }
  void push(Tri const &tri) {
    tris.push(tri);
    float3 tmin, tmax;
    tri.get_aabb(tmin, tmax);
    ito(3) min[i] = MIN(min[i], tmin[i]);
    ito(3) max[i] = MAX(max[i], tmax[i]);
  }
  void split(u32 max_items, u32 depth = 0) {
    ASSERT_DEBUG(depth < BVH_Node::MAX_DEPTH);
    if (tris.size > max_items && depth < BVH_Node::MAX_DEPTH) {
      left = new BVH_Helper;
      left->init();
      left->reserve(tris.size / 2);
      right = new BVH_Helper;
      right->init();
      right->reserve(tris.size / 2);
      struct Sorting_Node {
        u32   id;
        float val;
      };
      {
        TMP_STORAGE_SCOPE;
        u32           num_items = tris.size;
        Sorting_Node *sorted_dims[6];
        ito(6) sorted_dims[i] =
            (Sorting_Node *)tl_alloc_tmp(sizeof(Sorting_Node) * num_items);
        Tri *items = tris.ptr;
        ito(num_items) {
          float3 tmin, tmax;
          items[i].get_aabb(tmin, tmax);
          jto(3) {
            sorted_dims[j][i].val     = tmin[j];
            sorted_dims[j][i].id      = i;
            sorted_dims[j + 3][i].val = tmax[j];
            sorted_dims[j + 3][i].id  = i;
          }
        }
        ito(6) quicky_sort(sorted_dims[i], num_items,
                           [](Sorting_Node const &a, Sorting_Node const &b) {
                             return a.val < b.val;
                           });
        float max_dim_diff = 0.0f;
        u32   max_dim_id   = 0;
        u32   last_item    = num_items - 1;
        ito(3) {
          // max - min
          float diff =
              sorted_dims[i + 3][last_item].val - sorted_dims[i][0].val;
          ASSERT_DEBUG(diff > 0.0f);
          if (diff > max_dim_diff) {
            max_dim_diff = diff;
            max_dim_id   = i;
          }
        }
        u32 split_index = (last_item + 1) / 2;
        ito(num_items) {
          u32 tri_id = sorted_dims[max_dim_id][i].id;
          Tri tri    = tris[tri_id];
          if (i < split_index) {
            left->push(tri);
          } else {
            right->push(tri);
          }
        }
      }
      is_leaf = false;
      tris.release();
      left->split(max_items, depth + 1);
      right->split(max_items, depth + 1);
    }
  }
};

static_assert(sizeof(BVH_Node) == 28, "Blamey!");

struct BVH {
  static constexpr u32 MAX_DEPTH = 16;
  Pool<Tri>            tri_pool;
  Pool<BVH_Node>       node_pool;

  Tri *     tris;
  BVH_Node *root;

  void gen(BVH_Node *node, BVH_Helper *hnode) {
    ASSERT_ALWAYS(node != NULL);
    ASSERT_ALWAYS(hnode != NULL);
    if (hnode->is_leaf) {
      ASSERT_DEBUG(hnode->tris.size != 0);
      node->init_leaf(hnode->min, hnode->max, alloc_tri_chunk());
      ASSERT_DEBUG(hnode->tris.size <= BVH_Node::MAX_ITEMS);
      node->set_num_items(hnode->tris.size);
      Tri *tris = tri_pool.at(node->items_offset());
      ito(hnode->tris.size) { tris[i] = hnode->tris[i]; }
    } else {
      BVH_Node *children = node_pool.alloc(2);
      node->init_branch(hnode->min, hnode->max, children);
      gen(children + 0, hnode->left);
      gen(children + 1, hnode->right);
    }
  }
  void init(Tri *tris, u32 num_tris) { //
    BVH_Helper *hroot = new BVH_Helper;
    hroot->init();
    hroot->reserve(num_tris);
    defer(hroot->release());
    ito(num_tris) { hroot->push(tris[i]); }
    hroot->split(BVH_Node::MAX_ITEMS);
    tri_pool  = Pool<Tri>::create(1 << 24);
    node_pool = Pool<BVH_Node>::create(1 << 22);
    root      = node_pool.alloc(1);
    gen(root, hroot);
  }
  u32 alloc_tri_chunk() {
    Tri *tri_root  = tri_pool.at(0);
    Tri *new_chunk = tri_pool.alloc(BVH_Node::MAX_ITEMS);
    return (u32)(((u8 *)new_chunk - (u8 *)tri_root) / sizeof(Tri));
  }
  void release() {
    tri_pool.release();
    node_pool.release();
  }
  template <typename F> void traverse(float3 ro, float3 rd, F fn) {
    if (!root->intersects_ray(ro, rd)) return;
    traverse(root, ro, rd, fn);
  }
  template <typename F>
  void traverse(BVH_Node *node, float3 ro, float3 rd, F fn) {
    if (node->is_leaf()) {
      Tri *tris     = tri_pool.at(node->items_offset());
      u32  num_tris = node->num_items();
      ito(num_tris) fn(tris[i]);
    } else {
      BVH_Node *children = node->first_child();
      BVH_Node *left     = children + 0;
      BVH_Node *right    = children + 1;
      if (left->intersects_ray(ro, rd)) traverse(left, ro, rd, fn);
      if (right->intersects_ray(ro, rd)) traverse(right, ro, rd, fn);
    }
  }
};

void vec_test() {
  vfloat3 a = vfloat3::splat(1.0f, 2.0f, 3.0f);
  vfloat3 b = vfloat3::splat(2.0f, 3.0f, 5.0f);
  vfloat3 c = (1.0f + a + b * 2.0f).normalize();
  c.dump();
}

void sort_test() {
  int arr[] = {
      27, 12,  3,  32, 46, 97, 32, 60, 56, 91, 69, 76, 7,  95, 25, 86, 96,
      5,  88,  88, 42, 30, 35, 74, 93, 28, 82, 23, 98, 31, 22, 55, 53, 23,
      45, 78,  46, 97, 34, 63, 60, 91, 99, 58, 73, 53, 75, 63, 88, 32, 66,
      13, 100, 44, 22, 37, 23, 29, 70, 51, 51, 76, 66, 15, 39, 48, 85, 13,
      89, 97,  17, 36, 41, 75, 92, 43, 29, 82, 31, 59, 67, 26, 49, 38, 20,
      2,  98,  70, 31, 9,  79, 48, 11, 4,  44, 1,  49, 73, 90, 70,
  };
  kto(1000) {
    quicky_sort(arr, ARRAYSIZE(arr), [](int a, int b) { return a < b; });
    ito(ARRAYSIZE(arr) - 1) { ASSERT_ALWAYS(arr[i] <= arr[i + 1]); }
    jto(ARRAYSIZE(arr)) arr[j] = rand();
  }
}

struct Camera {
  float3 position;
  float3 look;
  float3 up;
  float3 right;
  float  fov;
};
Camera gen_camera(float phi, float theta, float r, float3 lookat, float fov) {
  Camera cam;
  cam.position =
      r * float3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
  cam.look = -normalize(cam.position);
  cam.position += lookat;
  cam.right = -normalize(cross(cam.look, float3(0.0, 1.0, 0.0)));
  cam.up    = -cross(cam.right, cam.look);
  cam.fov   = fov;
  return cam;
}
Ray gen_ray(Camera cam, float2 uv) {
  Ray r;
  r.o = cam.position;
  r.d = normalize(cam.look + cam.fov * (cam.right * uv.x + cam.up * uv.y));
  return r;
}

// Poor man's queue
// Not thread safe in all scenarios but kind of works in mine
// @Cleanup
template <typename Job_t> struct Queue {
  Array<Job_t>     job_queue;
  std::atomic<u32> head = 0;
  std::mutex       mutex;
  void             init() {
    job_queue.init();
    job_queue.resize(512 * 512 * 256);
  }
  void  release() { job_queue.release(); }
  Job_t dequeue() {
    ASSERT_PANIC(head);
    u32  old_head = head.fetch_sub(1);
    auto back     = job_queue[old_head - 1];

    return back;
  }
  // called in a single thread
  void dequeue(Job_t *out, u32 &count) {
    if (head < count) {
      count = head;
    }
    u32 old_head = head.fetch_sub(count);
    memcpy(out, &job_queue[head], count * sizeof(out[0]));
  }
  void enqueue(Job_t job) {
    //      std::scoped_lock<std::mutex> sl(mutex);
    ASSERT_PANIC(!std::isnan(job.ray_dir.x) && !std::isnan(job.ray_dir.y) &&
                 !std::isnan(job.ray_dir.z));
    u32 old_head        = head.fetch_add(1);
    job_queue[old_head] = job;
    ASSERT_PANIC(head < job_queue.size);
  }
  void enqueue(Job_t const *jobs, u32 num) {
    u32 old_head = head.fetch_add(num);
    ASSERT_PANIC(head < job_queue.size);
    memcpy(&job_queue[old_head], jobs, num * sizeof(Job_t));
  }
  bool has_job() { return head != 0u; }
  void reset() { head = 0u; }
};

template <typename Job_t> struct Job_System {
  struct JobDesc {
    u32 offset, size;
  };
  using JobFunc = std::function<void(JobDesc)>;
  struct JobPayload {
    JobFunc func;
    JobDesc desc;
  };

  u32               jobs_per_item     = 8 * 32 * 1000;
  bool              use_jobs          = true;
  u32               max_jobs_per_iter = 16 * 16 * 32 * 1000;
  Array<JobPayload> work;
  Array<Job_t>      cur_work;
  Queue<Job_t>      queue;

  template <typename F> void flush(F fn) {
    while (queue.has_job()) {
      iter(fn);
    }
  }

  void exec_queue() {
    marl::WaitGroup wg(work.size);
    for (u32 i = 0; i < work.size; i++) {
      marl::schedule([=] {
        defer(wg.done());
        auto item = work[i];
        item.func(item.desc);
      });
    }
    wg.wait();
  }

  template <typename F> void iter(F fn) {
    cur_work.resize(max_jobs_per_iter);
    u32 jobs_this_iter = max_jobs_per_iter;
    queue.dequeue(&cur_work[0], jobs_this_iter);
    work.reserve((jobs_this_iter + jobs_per_item - 1) / jobs_per_item);
    ito((jobs_this_iter + jobs_per_item - 1) / jobs_per_item) {
      JobPayload pl;
      pl.func = [this, fn](JobDesc const &desc) {
        fn(queue, &cur_work[desc.offset], desc.size);
      };
      pl.desc =
          JobDesc{i * jobs_per_item,
                  MIN(u32(jobs_this_iter) - i * jobs_per_item, jobs_per_item)};
      work.push(pl);
    }
    exec_queue();
  }

  void init() {
    cur_work.init();
    queue.init();
    work.init();
  }

  void release() {
    queue.release();
    cur_work.release();
    work.release();
  }
};

struct Scene {
  // 3D model + materials
  PBR_Model   model;
  Array<BVH>  bvhs;
  Pool<Tri>   tri_pool;
  Image2D_Raw env_spheremap;

  // Array<float4> normal_rt;
  // Array<float4> albedo_rt;

  void init(string_ref filename, string_ref env_filename) {

    model         = load_gltf_pbr(filename);
    env_spheremap = load_image(env_filename);
    bvhs.init();
    tri_pool = Pool<Tri>::create(1 << 20);
    kto(model.meshes.size) {
      Raw_Mesh_Opaque &mesh = model.meshes[k];
      tri_pool.reset();
      Tri *tris     = tri_pool.alloc(mesh.num_indices / 3);
      u32  num_tris = mesh.num_indices / 3;
      ito(num_tris) {
        Triangle_Full ftri = mesh.fetch_triangle(i);
        tris[i].a          = ftri.v0.position;
        tris[i].b          = ftri.v1.position;
        tris[i].c          = ftri.v2.position;
        tris[i].id         = i;
      }
      BVH bvh;
      bvh.init(tris, num_tris);
      bvhs.push(bvh);
    }
  }

  float4 env_value(float3 ray_dir, float3 color) {
    if (env_spheremap.data == NULL) {
      return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    float  theta = std::acos(ray_dir.y);
    float2 xy    = normalize(float2(ray_dir.z, -ray_dir.x));
    float  phi   = -std::atan2(xy.x, xy.y);
    return float4(color, 1.0f) *
           env_spheremap.sample(float2((phi / PI / 2.0f) + 0.5f, theta / PI));
  };

  bool collide(float3 ro, float3 rd, Collision &col) {
    col.t    = FLT_MAX;
    bool hit = false;
    kto(model.meshes.size) {
      BVH &bvh = bvhs[k];
      bvh.traverse(ro, rd, [&](Tri &tri) {
        Collision c;
        if (ray_triangle_test_moller(ro, rd, tri.a, tri.b, tri.c, c)) {
          if (c.t < col.t) {
            hit         = true;
            col         = c;
            col.mesh_id = k;
            col.face_id = tri.id;
          }
        }
      });
    }
    return hit;
    /*if (!hit) {
      float4 env_val = env_value(rd, float3(1.0f, 1.0f, 1.0f));
      return float3(env_val.x, env_val.y, env_val.z);
    } else {
      PBR_Material &mat = model.materials[col.mesh_id];
      const u32     N   = 16 >> depth;
      float3        res = float3(0.0f, 0.0f, 0.0f);
      ito(N) {
        float3    rn = rf.sample_lambert_BRDF(col.normal);
        Collision new_col;
        res +=
            trace(col.position + 1.0e-4f * col.normal, rn, new_col, depth + 1);
      }
      return res / float(N);
    }*/
  }

  void release() {
    ito(bvhs.size) bvhs[i].release();
    bvhs.release();
    env_spheremap.release();
    tri_pool.release();
    model.release();
  }
};

uint32_t rgba32f_to_rgba8_unorm(float r, float g, float b, float a) {
  uint8_t r8 = (uint8_t)(clamp(r, 0.0f, 1.0f) * 255.0f);
  uint8_t g8 = (uint8_t)(clamp(g, 0.0f, 1.0f) * 255.0f);
  uint8_t b8 = (uint8_t)(clamp(b, 0.0f, 1.0f) * 255.0f);
  uint8_t a8 = (uint8_t)(clamp(a, 0.0f, 1.0f) * 255.0f);
  return                     //
      ((uint32_t)r8 << 0) |  //
      ((uint32_t)g8 << 8) |  //
      ((uint32_t)b8 << 16) | //
      ((uint32_t)a8 << 24);  //
}

uint32_t rgba32f_to_srgba8_unorm(float r, float g, float b, float a) {
  uint8_t r8 = (uint8_t)(clamp(std::pow(r, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f);
  uint8_t g8 = (uint8_t)(clamp(std::pow(g, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f);
  uint8_t b8 = (uint8_t)(clamp(std::pow(b, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f);
  uint8_t a8 = (uint8_t)(clamp(std::pow(a, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f);
  return                     //
      ((uint32_t)r8 << 0) |  //
      ((uint32_t)g8 << 8) |  //
      ((uint32_t)b8 << 16) | //
      ((uint32_t)a8 << 24);  //
}

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  // vec_test();
  // sort_test();
  Scene scene;
   scene.init(stref_s("models/human_bust_sculpt/scene.gltf"),
  //scene.init(stref_s("models/tree_low-poly_3d_model/scene.gltf"),
             stref_s("env/lythwood_field_2k.hdr"));
  struct Path_Tracing_Job {
    float3 ray_origin;
    float3 ray_dir;
    // Color weight applied to the sampled light
    vec3 color;
    u32  pixel_x, pixel_y;
    f32  weight;
    // For visibility checks
    u32 light_id;
    u32 depth,
        // Used to track down bugs
        _depth;
  };
  Job_System<Path_Tracing_Job> js;
  Array<float4>                rt0;
  marl::Scheduler::Config      cfg;
  int2                         iResolution = int2(512, 512);
  cfg.setWorkerThreadCount(marl::Thread::numLogicalCPUs());
  marl::Scheduler scheduler(cfg);
  scheduler.bind();
  js.init();
  rt0.init();
  rt0.resize(iResolution.x * iResolution.y);
  rt0.memzero();
  auto trace_primary = [&](u32 i, u32 j, float3 ro, float3 rd) {
    Path_Tracing_Job job;
    job.color      = float3(1.0f, 1.0f, 1.0f);
    job.depth      = 0;
    job._depth     = 0;
    job.light_id   = 0;
    job.pixel_x    = j;
    job.pixel_y    = i;
    job.ray_dir    = rd;
    job.ray_origin = ro;
    job.weight     = 1.0f;
    js.queue.enqueue(job);
  };
  Array<vec3>      ray_dirs;
  Array<vec3>      ray_origins;
  Array<Collision> ray_collisions;
  Random_Factory   rf[0x100];
  ray_dirs.init();
  ray_origins.init();
  ray_collisions.init();
  defer({
    rt0.release();
    js.release();
    scheduler.unbind();
    ray_dirs.release();
    ray_origins.release();
    ray_collisions.release();
    scene.release();
  });

  float2      m  = float2(0.0f, 0.0f);
  const float PI = 3.141592654f;
  Camera      cam =
      gen_camera(PI * 0.3f, PI * 0.5, 45.0, float3(0.0, 10.0, 0.0), 1.0);
  {
    TMP_STORAGE_SCOPE;
    u8 * rgb_image = (u8 *)tl_alloc_tmp(iResolution.x * iResolution.y * 3);
    auto retire    = [&](u32 i, u32 j, float3 d) {
      u32 rgba8 = rgba32f_to_rgba8_unorm(d.r, d.g, d.b, 1.0f);
      rgb_image[i * iResolution.x * 3 + j * 3 + 0] = (rgba8 >> 0) & 0xffu;
      rgb_image[i * iResolution.x * 3 + j * 3 + 1] = (rgba8 >> 8) & 0xffu;
      rgb_image[i * iResolution.x * 3 + j * 3 + 2] = (rgba8 >> 16) & 0xffu;
    };
    //while (true) {
      ito(iResolution.y) {
        jto(iResolution.x) {
          float2 uv =
              float2((float(j) + 0.5f) / iResolution.y,
                     (float(iResolution.y - i - 1) + 0.5f) / iResolution.y) *
                  2.0f -
              1.0f;
          Ray ray        = gen_ray(cam, uv);
          u8  intersects = 0;
          trace_primary(i, j, ray.o, ray.d);
        }
      }
      js.flush([&](Queue<Path_Tracing_Job> &queue, Path_Tracing_Job *jobs,
                   u32 size) {
        // ray_dirs.resize(jobs_this_iter);
        // ray_origins.resize(jobs_this_iter);
        // ray_collisions.resize(jobs_this_iter);
        /* ito(jobs_this_iter) {
           ray_dirs[i]         = cur_work[i].ray_dir;
           ray_origins[i]      = cur_work[i].ray_origin;
           ray_collisions[i].t = FLT_MAX;
         }*/
        ZoneScoped;
        ito(size) {
          Path_Tracing_Job job = jobs[i];
          Collision        col;
          bool   collide = scene.collide(job.ray_origin, job.ray_dir, col);
          float3 result;
          if (collide) {
            result = float3(1.0f, 0.0f, 0.0f);
          } else {
            result = float3(0.0f, 0.0f, 0.0f);
          }
          //rt0[iResolution.x * job.pixel_y + job.pixel_x] +=
              //float4(result.x, result.y, result.z, 1.0f);
           retire(job.pixel_y, job.pixel_x, result);
        }
      });
    //}
     write_image_2d_i24_ppm("image.ppm", rgb_image, iResolution.x * 3,
                           iResolution.x, iResolution.y);
  }
#ifdef UTILS_TL_IMPL_DEBUG
  assert_tl_alloc_zero();
#endif
  return 0;
}