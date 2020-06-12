#define UTILS_IMPL
//#define UTILS_AVX512
#include "utils.hpp"

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include <assimp/contrib/stb_image/stb_image.h>
#include <assimp/pbrmaterial.h>

using namespace glm;

using int2     = ivec2;
using int3     = ivec3;
using int4     = ivec4;
using uint2    = uvec2;
using uint3    = uvec3;
using uint4    = uvec4;
using float2   = vec2;
using float3   = vec3;
using float4   = vec4;
using float2x2 = mat2;
using float3x3 = mat3;
using float4x4 = mat4;

#define PACK_SIZE 16

enum class Format_t {
  RGBA8_UNORM = 0,
  RGBA8_SRGB,
  RGB8_UNORM,
  RG32_FLOAT,
  RGB32_FLOAT,
  RGBA32_FLOAT,
};

struct Image2D_Raw {
  u32      width;
  u32      height;
  Format_t format;
  u8 *     data;
  void     init(u32 width, u32 height, Format_t format, u8 *data) {
    MEMZERO(*this);
    this->width  = width;
    this->height = height;
    this->format = format;
    u32 size     = get_bpp() * width * height;
    this->data   = (u8 *)malloc(size);
    memcpy(this->data, data, size);
  }
  void release() {
    if (data != NULL) free(data);
    MEMZERO(*this);
  }
  u32 get_bpp() {
    switch (format) {
    case Format_t::RGBA8_UNORM:
    case Format_t::RGBA8_SRGB: return 4u;
    case Format_t::RGB32_FLOAT: return 12u;
    default: ASSERT_PANIC(false && "unsupported format");
    }
  }
  vec4 load(uint2 coord) {
    u32 bpc = 4u;
    switch (format) {
    case Format_t::RGBA8_UNORM:
    case Format_t::RGBA8_SRGB: bpc = 4u; break;
    case Format_t::RGB32_FLOAT: bpc = 12u; break;
    default: ASSERT_PANIC(false && "unsupported format");
    }
    auto load_f32 = [&](uint2 coord, u32 component) {
      uint2 size = uint2(width, height);
      return *(
          f32 *)&data[coord.x * bpc + coord.y * size.x * bpc + component * 4u];
    };
    uint2 size = uint2(width, height);
    if (coord.x >= size.x) coord.x = size.x - 1;
    if (coord.y >= size.y) coord.y = size.y - 1;
    switch (format) {
    case Format_t::RGBA8_UNORM: {
      u8 r = data[coord.x * bpc + coord.y * size.x * bpc];
      u8 g = data[coord.x * bpc + coord.y * size.x * bpc + 1u];
      u8 b = data[coord.x * bpc + coord.y * size.x * bpc + 2u];
      u8 a = data[coord.x * bpc + coord.y * size.x * bpc + 3u];
      return vec4(float(r) / 255.0f, float(g) / 255.0f, float(b) / 255.0f,
                  float(a) / 255.0f);
    }
    case Format_t::RGBA8_SRGB: {
      u8 r = data[coord.x * bpc + coord.y * size.x * bpc];
      u8 g = data[coord.x * bpc + coord.y * size.x * bpc + 1u];
      u8 b = data[coord.x * bpc + coord.y * size.x * bpc + 2u];
      u8 a = data[coord.x * bpc + coord.y * size.x * bpc + 3u];

      auto out = vec4(float(r) / 255.0f, float(g) / 255.0f, float(b) / 255.0f,
                      float(a) / 255.0f);
      out.r    = std::pow(out.r, 2.2f);
      out.g    = std::pow(out.g, 2.2f);
      out.b    = std::pow(out.b, 2.2f);
      out.a    = std::pow(out.a, 2.2f);
      return out;
    }
    case Format_t::RGB32_FLOAT: {
      f32 r = load_f32(coord, 0u);
      f32 g = load_f32(coord, 1u);
      f32 b = load_f32(coord, 2u);
      return vec4(r, g, b, 1.0f);
    }
    default: ASSERT_PANIC(false && "unsupported format");
    }
  };
  vec4 sample(vec2 uv) {
    ivec2 size    = ivec2(width, height);
    vec2  suv     = uv * vec2(float(size.x - 1u), float(size.y - 1u));
    ivec2 coord[] = {
        ivec2(i32(suv.x), i32(suv.y)),
        ivec2(i32(suv.x), i32(suv.y + 1.0f)),
        ivec2(i32(suv.x + 1.0f), i32(suv.y)),
        ivec2(i32(suv.x + 1.0f), i32(suv.y + 1.0f)),
    };
    ito(4) {
      // Repeat
      jto(2) {
        while (coord[i][j] >= size[j]) coord[i][j] -= size[j];
        while (coord[i][j] < 0) coord[i][j] += size[j];
      }
    }
    vec2  fract = vec2(suv.x - std::floor(suv.x), suv.y - std::floor(suv.y));
    float weights[] = {
        (1.0f - fract.x) * (1.0f - fract.y),
        (1.0f - fract.x) * (fract.y),
        (fract.x) * (1.0f - fract.y),
        (fract.x) * (fract.y),
    };
    vec4 result = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    ito(4) result += load(uint2(coord[i].x, coord[i].y)) * weights[i];
    return result;
  };
};

enum class Index_t { U32, U16 };

enum class Attribute_t {
  NONE = 0,
  POSITION,
  NORMAL,
  BINORMAL,
  TANGENT,
  UV0,
  UV1,
  UV2,
  UV3
};

struct Vertex_Full {
  float3 pos;
  float3 normal;
  float3 binormal;
  float3 tangent;
  float2 u0;
  float2 u1;
  float2 u2;
  float2 u3;
};
struct Tri_Index {
  u32 i0, i1, i2;
};
struct Triangle_Full {
  Vertex_Full v0;
  Vertex_Full v1;
  Vertex_Full v2;
};
// We are gonna use one simplified material schema for everything
struct PBR_Material {
  // R8G8B8A8
  i32 normal_id = -1;
  // R8G8B8A8
  i32 albedo_id = -1;
  // R8G8B8A8
  // AO+Roughness+Metalness
  i32    arm_id           = -1;
  f32    metal_factor     = 1.0f;
  f32    roughness_factor = 1.0f;
  float4 albedo_factor    = float4(1.0f);
};

struct Raw_Mesh_Opaque {
  struct Attribute {
    Attribute_t type;
    Format_t    format;
    u32         offset;
  };
  Array<u8>                  attribute_data;
  Array<u8>                  index_data;
  Index_t                    index_type;
  InlineArray<Attribute, 16> attributes;
  u32                        vertex_stride;

  Raw_Mesh_Opaque() { init(); }
  ~Raw_Mesh_Opaque() { release(); }

  void init() {
    attributes.init();
    attribute_data.init();
    index_data.init();
    vertex_stride = 0;
  }
  void release() {
    attributes.release();
    attribute_data.release();
    index_data.release();
    vertex_stride = 0;
  }

  float3 fetch_position(u32 index) {
    ito(attributes.size) {
      switch (attributes[i].type) {

      case Attribute_t::POSITION:
        ASSERT_PANIC(attributes[i].format == Format_t::RGB32_FLOAT);
        float3 pos;
        memcpy(&pos,
               attribute_data.at(index * vertex_stride + attributes[i].offset),
               12);
        return pos;
      default: break;
      }
    }
    TRAP;
  }
  Vertex_Full fetch_vertex(u32 index) {
    Vertex_Full v;
    MEMZERO(v);
    ito(attributes.size) {
      switch (attributes[i].type) {
      case Attribute_t::NORMAL:
        ASSERT_PANIC(attributes[i].format == Format_t::RGB32_FLOAT);
        memcpy(&v.normal,
               attribute_data.at(index * vertex_stride + attributes[i].offset),
               12);
        break;
      case Attribute_t::BINORMAL:
        ASSERT_PANIC(attributes[i].format == Format_t::RGB32_FLOAT);
        memcpy(&v.binormal,
               attribute_data.at(index * vertex_stride + attributes[i].offset),
               12);
        break;
      case Attribute_t::TANGENT:
        ASSERT_PANIC(attributes[i].format == Format_t::RGB32_FLOAT);
        memcpy(&v.tangent,
               attribute_data.at(index * vertex_stride + attributes[i].offset),
               12);
        break;
      case Attribute_t::POSITION:
        ASSERT_PANIC(attributes[i].format == Format_t::RGB32_FLOAT);
        memcpy(&v.pos,
               attribute_data.at(index * vertex_stride + attributes[i].offset),
               12);
        break;
      case Attribute_t::UV0:
        ASSERT_PANIC(attributes[i].format == Format_t::RG32_FLOAT);
        memcpy(&v.u0,
               attribute_data.at(index * vertex_stride + attributes[i].offset),
               8);
        break;
      case Attribute_t::UV1:
        ASSERT_PANIC(attributes[i].format == Format_t::RG32_FLOAT);
        memcpy(&v.u1,
               attribute_data.at(index * vertex_stride + attributes[i].offset),
               8);
        break;
      case Attribute_t::UV2:
        ASSERT_PANIC(attributes[i].format == Format_t::RG32_FLOAT);
        memcpy(&v.u2,
               attribute_data.at(index * vertex_stride + attributes[i].offset),
               8);
        break;
      case Attribute_t::UV3:
        ASSERT_PANIC(attributes[i].format == Format_t::RG32_FLOAT);
        memcpy(&v.u3,
               attribute_data.at(index * vertex_stride + attributes[i].offset),
               8);
        break;
      default: TRAP;
      }
    }
    return v;
  }

  Tri_Index get_tri_index(u32 id) {
    Tri_Index o;
    if (index_type == Index_t::U16) {
      o.i0 = (u32) * (u16 *)index_data.at(id * 2 + 0);
      o.i1 = (u32) * (u16 *)index_data.at(id * 2 + 1);
      o.i2 = (u32) * (u16 *)index_data.at(id * 2 + 2);
    } else {
      o.i0 = (u32) * (u32 *)index_data.at(id * 4 + 0);
      o.i1 = (u32) * (u32 *)index_data.at(id * 4 + 1);
      o.i2 = (u32) * (u32 *)index_data.at(id * 4 + 2);
    }
    return o;
  }

  Triangle_Full fetch_triangle(u32 id) {
    Tri_Index   tind = get_tri_index(id);
    Vertex_Full v0   = fetch_vertex(tind.i0);
    Vertex_Full v1   = fetch_vertex(tind.i1);
    Vertex_Full v2   = fetch_vertex(tind.i2);
    return {v0, v1, v2};
  }
};

// https://github.com/graphitemaster/normals_revisited
static float minor(const float m[16], int r0, int r1, int r2, int c0, int c1,
                   int c2) {
  return m[4 * r0 + c0] * (m[4 * r1 + c1] * m[4 * r2 + c2] -
                           m[4 * r2 + c1] * m[4 * r1 + c2]) -
         m[4 * r0 + c1] * (m[4 * r1 + c0] * m[4 * r2 + c2] -
                           m[4 * r2 + c0] * m[4 * r1 + c2]) +
         m[4 * r0 + c2] * (m[4 * r1 + c0] * m[4 * r2 + c1] -
                           m[4 * r2 + c0] * m[4 * r1 + c1]);
}

static void cofactor(const float src[16], float dst[16]) {
  dst[0]  = minor(src, 1, 2, 3, 1, 2, 3);
  dst[1]  = -minor(src, 1, 2, 3, 0, 2, 3);
  dst[2]  = minor(src, 1, 2, 3, 0, 1, 3);
  dst[3]  = -minor(src, 1, 2, 3, 0, 1, 2);
  dst[4]  = -minor(src, 0, 2, 3, 1, 2, 3);
  dst[5]  = minor(src, 0, 2, 3, 0, 2, 3);
  dst[6]  = -minor(src, 0, 2, 3, 0, 1, 3);
  dst[7]  = minor(src, 0, 2, 3, 0, 1, 2);
  dst[8]  = minor(src, 0, 1, 3, 1, 2, 3);
  dst[9]  = -minor(src, 0, 1, 3, 0, 2, 3);
  dst[10] = minor(src, 0, 1, 3, 0, 1, 3);
  dst[11] = -minor(src, 0, 1, 3, 0, 1, 2);
  dst[12] = -minor(src, 0, 1, 2, 1, 2, 3);
  dst[13] = minor(src, 0, 1, 2, 0, 2, 3);
  dst[14] = -minor(src, 0, 1, 2, 0, 1, 3);
  dst[15] = minor(src, 0, 1, 2, 0, 1, 2);
}

static float4x4 cofactor(float4x4 const &in) {
  float4x4 out;
  cofactor(&in[0][0], &out[0][0]);
  return out;
}

struct Transform_Node {
  float3     offset;
  quat       rotation;
  float      scale;
  float4x4   transform_cache;
  Array<u32> meshes;
  Array<u32> children;
  void       init() {
    MEMZERO(*this);
    meshes.init();
    children.init();
    scale           = 1.0f;
    transform_cache = float4x4(1.0f);
  }
  void release() {
    meshes.release();
    children.release();
    MEMZERO(*this);
  }
  void update_cache(float4x4 const &parent = float4x4(1.0f)) {
    transform_cache = parent * get_transform();
  }
  float4x4 get_transform() {
    //  return transform;
    return glm::translate(float4x4(1.0f), offset) * (float4x4)rotation *
           glm::scale(float4x4(1.0f), float3(scale, scale, scale));
  }
  float4x4 get_cofactor() {
    mat4 out{};
    mat4 transform = get_transform();
    cofactor(&transform[0][0], &out[0][0]);
  }
};

// To make things simple we use one format of meshes
struct PBR_Model {
  Array<Image2D_Raw>     images;
  Array<Raw_Mesh_Opaque> meshes;
  Array<PBR_Material>    materials;
  Array<Transform_Node>  nodes;

  void init() {
    images.init();
    meshes.init();
    materials.init();
    nodes.init();
  }
  void release() {
    images.release();
    meshes.release();
    materials.release();
    nodes.release();
  }
};

struct Collision {
  vfloat3 pos;
  vfloat3 norm;
  vfloat  t;
};

struct Ray {
  vfloat3 o;
  vfloat3 d;
};
#if 0
				
struct KDNode {
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
  // +->+-------------+----------------------------+
  //    | [31:30] dim | [24:1]  First child offset |
  //    +-------------+----------------------------+

  // constants
  static constexpr u32 LEAF_BIT  = 1 << 31;
  static constexpr u32 DIM_BITS  = 0b11; // [00 - x, 01 - y, 10 - z, 11 - undef]
  static constexpr u32 DIM_SHIFT = 29;
  // Leaf flags:
  // 1 + 7 + 24 = 32 bits used
  static constexpr u32 ITEMS_OFFSET_MASK  = 0xffffff;  // 24 bits
  static constexpr u32 ITEMS_OFFSET_SHIFT = 0;         // low bits
  static constexpr u32 NUM_ITEMS_MASK     = 0b1111111; // 7 bits
  static constexpr u32 NUM_ITEMS_SHIFT    = 24;        // after first 24 bits
  static constexpr u32 MAX_ITEMS          = 128;       // max items
  // Node flags:
  // 1 + 2 + 24 = 28 bits used
  static constexpr u32 FIRST_CHILD_MASK  = 0xffffff;
  static constexpr u32 FIRST_CHILD_SHIFT = 0;

  // payload
  u32 flags;
  f32 bias;

  // methods
  float3 get_dir() {
    u8 dim_bits = ((flags >> DIM_SHIFT) & DIM_BITS);
    if (dim_bits == 0b00) {
      return float3(1.0f, 0.0f, 0.0f);
    }
    if (dim_bits == 0b01) {
      return float3(0.0f, 1.0f, 0.0f);
    }
    if (dim_bits == 0b10) {
      return float3(0.0f, 0.0f, 1.0f);
    }
    TRAP;
  }
  void init_leaf(u32 offset) {
    flags = LEAF_BIT;
    ASSERT_DEBUG(offset < ITEMS_OFFSET_MASK);
    flags |= offset << ITEMS_OFFSET_SHIFT;
  }
  void init_branch(KDNode *node, u8 dim, float bias) {
    ptrdiff_t diff = ((u8 *)node - (u8 *)this) / sizeof(KDNode);
    ASSERT_DEBUG(diff > 0 && diff < FIRST_CHILD_MASK);
    flags = (diff << FIRST_CHILD_SHIFT);
    flags |= ((dim & DIM_BITS) << DIM_SHIFT);
    this->bias = bias;
  }
  bool is_leaf() { return (flags & LEAF_BIT) == LEAF_BIT; }
  u32  num_items() { return ((flags >> NUM_ITEMS_SHIFT) & NUM_ITEMS_MASK); }
  u32  items_offset() {
    return ((flags >> ITEMS_OFFSET_SHIFT) & ITEMS_OFFSET_MASK);
  }
  KDNode *first_child() {
    return this + (((flags >> FIRST_CHILD_SHIFT) & FIRST_CHILD_MASK));
  }
  void set_num_items(u32 num) {
    ASSERT_DEBUG(num <= NUM_ITEMS_MASK);
    flags &= ~(NUM_ITEMS_MASK << NUM_ITEMS_SHIFT);
    flags |= (num << NUM_ITEMS_SHIFT);
  }
  void add_item() { set_num_items(num_items() + 1); }
  bool is_full() { return num_items() == NUM_ITEMS_MASK; }
};

static_assert(sizeof(KDNode) == 8, "Blamey!");


struct KDTree {
  // constants
  static constexpr u32 MAX_DEPTH         = 16;
  static constexpr u32 MAX_TRIS_PER_LEAF = 32;

  // data
  Pool<KDTri>  tri_pool;
  Pool<KDNode> node_pool;

  KDTri * tris;
  KDNode *root;
  // methods
  u32 alloc_tri_chunk() {
    KDTri *tri_root  = tri_pool.at(0);
    KDTri *new_chunk = tri_pool.alloc(KDNode::MAX_ITEMS);
    return (u32)(((u8 *)new_chunk - (u8 *)tri_root) / sizeof(KDTri));
  }
  void split(KDNode *node, float3 node_dir, float node_bias, float3 node_min,
             float3 node_max, u32 depth) {
    ASSERT_DEBUG(depth < MAX_DEPTH);
    u32    items_offset = node->items_offset();
    KDTri *tris         = tri_pool.at(items_offset);
    u32    num_items    = node->num_items();
    TMP_STORAGE_SCOPE;
    KDTri *tmp_items = (KDTri *)tl_alloc_tmp(sizeof(KDTri) * num_items);
    memcpy(tmp_items, tris, sizeof(KDTri) * num_items);
    KDNode *children = node_pool.alloc(2);
    children[0].init_leaf(items_offset);
    children[1].init_leaf(alloc_tri_chunk());
    // dimensions
    kto(3) {
      // start points/end points
      float sp[KDNode::MAX_ITEMS];
      float ep[KDNode::MAX_ITEMS];
      ito(node->num_items()) { KDTri tri = tris[i]; }
    }
    // node->init_branch(children, );
    // Push items from the parent to its children
    // float child_size = node_size / 2.0f;
    /* ito(4) {
       Vec2 ch_pos = get_child_pos(node_pos, node_size, i);
       jto(num_items) {
         QuadItem item = tmp_items[j];
         if (intersects({item.pos_x, item.pos_y}, item.size, ch_pos,
                        child_size)) {
           push_item(children + i, ch_pos, child_size, {item.pos_x, item.pos_y},
                     item.size, item.id, depth + 1);
         }
       }
     }*/
  }
  void init() { //
    tri_pool  = Pool<KDTri>::create(1 << 20);
    node_pool = Pool<KDNode>::create(1 << 16);
    root      = node_pool.alloc(1);
    root->init_leaf(alloc_tri_chunk());
  }
  void push_tri(float3 a, float3 b, float3 c //
  ) {                                        //
  }
  void trace(Ray const &ray, Collision &col) { //
  }
};
#endif // 0

struct Tri {
  u32    id;
  float3 a;
  float3 b;
  float3 c;
  void   get_aabb(float3 &min, float3 &max) const {
    ito(i) min[i] = MIN(a[i], MIN(b[i], c[i]));
    ito(i) max[i] = MAX(a[i], MAX(b[i], c[i]));
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
  // +->+-------------+----------------------------+
  //    | [31:30] dim | [24:1]  First child offset |
  //    +-------------+----------------------------+

  // constants
  static constexpr u32 LEAF_BIT  = 1 << 31;
  static constexpr u32 DIM_BITS  = 0b11; // [00 - x, 01 - y, 10 - z, 11 - undef]
  static constexpr u32 DIM_SHIFT = 29;
  // Leaf flags:
  static constexpr u32 ITEMS_OFFSET_MASK  = 0xffffff;  // 24 bits
  static constexpr u32 ITEMS_OFFSET_SHIFT = 0;         // low bits
  static constexpr u32 NUM_ITEMS_MASK     = 0b1111111; // 7 bits
  static constexpr u32 NUM_ITEMS_SHIFT    = 24;        // after first 24 bits
  static constexpr u32 MAX_ITEMS          = 4;         // max items
  // Node flags:
  static constexpr u32 FIRST_CHILD_MASK  = 0xffffff;
  static constexpr u32 FIRST_CHILD_SHIFT = 0;

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
  void init_leaf(float3 min, float3 max, u32 offset) {
    flags = LEAF_BIT;
    ASSERT_DEBUG(offset < MAX_ITEMS);
    flags |= ((offset << ITEMS_OFFSET_SHIFT));
    this->min = min;
    this->max = max;
  }
  void init_branch(float3 min, float3 max, BVH_Node *child, u8 dim,
                   float bias) {
    ptrdiff_t diff = ((u8 *)child - (u8 *)this) / sizeof(BVH_Node);
    ASSERT_DEBUG(diff > 0 && diff < FIRST_CHILD_MASK);
    flags = ((u32)diff << FIRST_CHILD_SHIFT);
    flags |= ((dim & DIM_BITS) << DIM_SHIFT);
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
  void        init() {
    MEMZERO(*this);
    tris.init();
    min = float3(1.0e10f, 1.0e10f, 1.0e10f);
    max = float3(-1.0e10f, -1.0e10f, -1.0e10f);
  }
  void release() {
    if (left != NULL) left->release();
    if (right != NULL) right->release();
    tris.release();
    MEMZERO(*this);
    delete this;
  }
  void push(Tri const &tri) {
    tris.push(tri);
    float3 tmin, tmax;
    tri.get_aabb(tmin, tmax);
    ito(3) min[i] = MIN(min[i], tmin[i]);
    ito(3) max[i] = MAX(max[i], tmax[i]);
  }
  void split(u32 max_items) {
    if (tris.size >= max_items) {
      left = new BVH_Helper;
      left->init();
      right = new BVH_Helper;
      right->init();
      struct Sorting_Node {
        u32   id;
        float val;
      };
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
        float diff = sorted_dims[i + 3][last_item].val - sorted_dims[i][0].val;
        ASSERT_DEBUG(diff > 0.0f);
        if (diff > max_dim_diff) {
          max_dim_diff = diff;
          max_dim_id   = i;
        }
      }
      u32 split_index = last_item / 2;
      ito(num_items) {
        u32 tri_id = sorted_dims[max_dim_id][i].id;
        Tri tri    = tris[tri_id];
        if (i < split_index) {
          left->push(tri);
        } else {
          right->push(tri);
        }
      }
      left->split(max_items);
      right->split(max_items);
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
  void      init(Tri *tris, u32 num_tris) { //
    BVH_Helper *hroot = new BVH_Helper;
    hroot->init();
    defer(hroot->release());
    float3 min, max;
    ito(num_tris) { hroot->push(tris[i]); }
    hroot->split(BVH_Node::MAX_ITEMS);
    tri_pool  = Pool<Tri>::create(1 << 20);
    node_pool = Pool<BVH_Node>::create(1 << 16);
    root      = node_pool.alloc(1);
    root->init_leaf(min, max, alloc_tri_chunk());
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
  void split(BVH_Node *node) {

    // float split_min = 1.0e10f;
    // float split_max = -1.0e10f;
  }
  void push(BVH_Node *node, Tri const &tri, float3 const &tmin,
            float3 const &tmax) {
    if (node->is_leaf()) {
      if (node->is_full()) {
        split(node);
        goto is_branch;
      } else {
        u32  num_items  = node->num_items();
        Tri *tris       = tri_pool.at(node->items_offset());
        tris[num_items] = tri;
        node->add_item();
      }
    }
  is_branch:
    BVH_Node *child = node->first_child();
    ito(2) if (child[i].intersects(tmin, tmax)) {
      push(child + i, tri, tmin, tmax);
    }
  }
  void push(float3 a, float3 b, float3 c, u32 id) {
    Tri tri;
    tri.a  = a;
    tri.b  = b;
    tri.c  = c;
    tri.id = id;
    float3 tmin, tmax;
    tri.get_aabb(tmin, tmax);
  }
};

struct Triangle_Full {
  Vertex_Full v0;
  Vertex_Full v1;
  Vertex_Full v2;
};

struct Camera {};

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

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  vec_test();
  sort_test();
  return 0;
}

void calculate_dim(const aiScene *scene, aiNode *node, vec3 &min, vec3 &max) {
  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
      kto(3) max[k] = std::max(max[k], mesh->mVertices[i][k]);
      kto(3) min[k] = std::min(min[k], mesh->mVertices[i][k]);
    }
  }
  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    calculate_dim(scene, node->mChildren[i], min, max);
  }
}

void traverse_node(PBR_Model &out, aiNode *node, const aiScene *scene,
                   std::string const &dir, u32 parent_id, float vk) {
  Transform_Node tnode{};
  mat4           transform;
  ito(4) {
    jto(4) { transform[i][j] = node->mTransformation[j][i]; }
  }
  vec3 offset;
  vec3 scale;

  ito(3) {
    scale[i] =
        glm::length(vec3(transform[0][i], transform[1][i], transform[2][i]));
  }

  offset = vec3(transform[3][0], transform[3][1], transform[3][2]);

  mat3 rot_mat;

  ito(3) {
    jto(3) { rot_mat[i][j] = transform[i][j] / scale[i]; }
  }
  quat rotation(rot_mat);

  //  tnode.offset = offset;
  //  tnode.rotation = rotation;
  //    tnode.transform = transform;

  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    // No support for animated meshes
    ASSERT_PANIC(!mesh->HasBones());
    Raw_Mesh_Opaque opaque_mesh{};
    using GLRF_Vertex_t = GLRF_Vertex_Static;
    auto write_bytes    = [&](u8 *src, size_t size) {
      // f32 *debug = (f32*)src;
      ito(size) opaque_mesh.attributes.push_back(src[i]);
    };

    ////////////////////////
    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
      GLRF_Vertex_t vertex{};

      vertex.position.x = mesh->mVertices[i].x * vk;
      vertex.position.y = mesh->mVertices[i].y * vk;
      vertex.position.z = mesh->mVertices[i].z * vk;
      if (mesh->HasTangentsAndBitangents()) {
        vertex.tangent.x = mesh->mTangents[i].x;
        vertex.tangent.y = mesh->mTangents[i].y;
        vertex.tangent.z = mesh->mTangents[i].z;

      } else {
        vertex.tangent = vec3(0.0f);
      }
      if (mesh->HasTangentsAndBitangents()) {
        vertex.binormal.x = mesh->mBitangents[i].x;
        vertex.binormal.y = mesh->mBitangents[i].y;
        vertex.binormal.z = mesh->mBitangents[i].z;

      } else {
        vertex.binormal = vec3(0.0f);
      }

      vertex.normal.x = mesh->mNormals[i].x;
      vertex.normal.y = mesh->mNormals[i].y;
      vertex.normal.z = mesh->mNormals[i].z;

      // An attempt to fix the tangent space
      if (std::isnan(vertex.binormal.x) || std::isnan(vertex.binormal.y) ||
          std::isnan(vertex.binormal.z)) {
        vertex.binormal =
            glm::normalize(glm::cross(vertex.normal, vertex.tangent));
      }
      if (std::isnan(vertex.tangent.x) || std::isnan(vertex.tangent.y) ||
          std::isnan(vertex.tangent.z)) {
        vertex.tangent =
            glm::normalize(glm::cross(vertex.normal, vertex.binormal));
      }
      ASSERT_PANIC(!std::isnan(vertex.binormal.x) &&
                   !std::isnan(vertex.binormal.y) &&
                   !std::isnan(vertex.binormal.z));
      ASSERT_PANIC(!std::isnan(vertex.tangent.x) &&
                   !std::isnan(vertex.tangent.y) &&
                   !std::isnan(vertex.tangent.z));

      if (mesh->HasTextureCoords(0)) {
        vertex.texcoord.x = mesh->mTextureCoords[0][i].x;
        vertex.texcoord.y = mesh->mTextureCoords[0][i].y;
      } else {
        vertex.texcoord = glm::vec2(0.0f, 0.0f);
      }
      write_bytes((u8 *)&vertex, sizeof(vertex));
    }
    for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
      aiFace face = mesh->mFaces[i];
      for (unsigned int j = 0; j < face.mNumIndices; ++j) {
        opaque_mesh.indices.push_back(face.mIndices[j]);
      }
    }

    aiMaterial * material = scene->mMaterials[mesh->mMaterialIndex];
    PBR_Material out_material;
    float        metal_base;
    float        roughness_base;
    aiColor4D    albedo_base;
    material->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_BASE_COLOR_FACTOR,
                  albedo_base);
    material->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR,
                  metal_base);
    material->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR,
                  roughness_base);
    out_material.metal_factor     = metal_base;
    out_material.roughness_factor = roughness_base;
    out_material.albedo_factor =
        vec4(albedo_base.r, albedo_base.g, albedo_base.b, albedo_base.a);
    //    material->GetTexture();
    for (int tex = aiTextureType_NONE; tex <= aiTextureType_UNKNOWN; tex++) {
      aiTextureType type = static_cast<aiTextureType>(tex);

      if (material->GetTextureCount(type) > 0) {
        aiString relative_path;

        material->GetTexture(type, 0, &relative_path);
        std::string full_path;
        full_path         = dir + "/";
        full_path         = full_path.append(relative_path.C_Str());
        vk::Format format = vk::Format::eR8G8B8A8Srgb;

        switch (type) {
        case aiTextureType_NORMALS:
          format                 = vk::Format::eR8G8B8A8Unorm;
          out_material.normal_id = i32(out.images.size());
          break;
        case aiTextureType_DIFFUSE:
          out_material.albedo_id = i32(out.images.size());
          break;

        case aiTextureType_SPECULAR:
        case aiTextureType_SHININESS:
        case aiTextureType_REFLECTION:
        case aiTextureType_UNKNOWN:
          //        case aiTextureType_AMBIENT:
          // @Cleanup :(
          // Some models downloaded from sketchfab have metallic-roughness
          // imported as unknown/lightmap and have (ao, roughness, metalness)
          // as components
        case aiTextureType_LIGHTMAP:
          format              = vk::Format::eR8G8B8A8Unorm;
          out_material.arm_id = i32(out.images.size());
          break;
        default:
          std::cerr << "[LOAD][WARNING] Unrecognized image type: " << type
                    << " with full path: " << full_path << "\n";
          //          ASSERT_PANIC(false && "Unsupported texture type");
          break;
        }
        out.images.emplace_back(load_image(full_path, format));
      } else {
      }
    }
    opaque_mesh.vertex_stride = sizeof(GLRF_Vertex_t);
    opaque_mesh.binding       = {
        {"POSITION",
         {0, offsetof(GLRF_Vertex_t, position), vk::Format::eR32G32B32Sfloat}},
        {"NORMAL",
         {0, offsetof(GLRF_Vertex_t, normal), vk::Format::eR32G32B32Sfloat}},
        {"TANGENT",
         {0, offsetof(GLRF_Vertex_t, tangent), vk::Format::eR32G32B32Sfloat}},
        {"BINORMAL",
         {0, offsetof(GLRF_Vertex_t, binormal), vk::Format::eR32G32B32Sfloat}},
        {"TEXCOORD_0",
         {0, offsetof(GLRF_Vertex_t, texcoord), vk::Format::eR32G32Sfloat}},
    };
    out.materials.push_back(out_material);
    out.meshes.push_back(opaque_mesh);
    tnode.meshes.push_back(u32(out.meshes.size() - 1));
  }
  out.nodes.push_back(tnode);
  out.nodes[parent_id].children.push_back(u32(out.nodes.size() - 1));
  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    traverse_node(out, node->mChildren[i], scene, dir,
                  u32(out.nodes.size() - 1), vk);
  }
}

PBR_Model load_gltf_pbr(std::string const &filename) {
  Assimp::Importer importer;
  PBR_Model        out;
  out.nodes.push_back(Transform_Node{});
  std::filesystem::path p(filename);
  std::filesystem::path dir   = p.parent_path();
  const aiScene *       scene = importer.ReadFile(
      filename.c_str(),
      aiProcess_Triangulate |
          // @TODO: Find out why transforms are not handled correcly otherwise
          aiProcess_GenSmoothNormals | aiProcess_PreTransformVertices |
          aiProcess_OptimizeMeshes | aiProcess_CalcTangentSpace |
          aiProcess_FlipUVs);
  if (!scene) {
    std::cerr << "[FILE] Errors: " << importer.GetErrorString() << "\n";
    ASSERT_PANIC(false);
  }
  vec3 max = vec3(0.0f);
  vec3 min = vec3(0.0f);
  calculate_dim(scene, scene->mRootNode, min, max);
  vec3 max_dims = max - min;
  // @Cleanup
  // Size normalization hack
  float vk      = 1.0f;
  float max_dim = std::max(max_dims.x, std::max(max_dims.y, max_dims.z));
  vk            = 50.0f / max_dim;
  vec3 avg      = (max + min) / 2.0f;
  traverse_node(out, scene->mRootNode, scene, dir.string(), 0, vk);

  out.nodes[0].offset = -avg * vk;
  return out;
}

Image_Raw load_image(std::string const &filename, vk::Format format) {
  if (filename.find(".hdr") != std::string::npos) {
    int            width, height, channels;
    unsigned char *result;
    FILE *         f = stbi__fopen(filename.c_str(), "rb");
    ASSERT_PANIC(f);
    stbi__context s;
    stbi__start_file(&s, f);
    stbi__result_info ri;
    memset(&ri, 0, sizeof(ri));
    ri.bits_per_channel = 8;
    ri.channel_order    = STBI_ORDER_RGB;
    ri.num_channels     = 0;
    float *hdr = stbi__hdr_load(&s, &width, &height, &channels, STBI_rgb, &ri);

    fclose(f);
    ASSERT_PANIC(hdr)
    Image_Raw out;
    out.width  = width;
    out.height = height;
    out.format = vk::Format::eR32G32B32Sfloat;
    out.data.resize(width * height * 3u * 4u);
    memcpy(&out.data[0], hdr, out.data.size());
    stbi_image_free(hdr);
    return out;
  } else {
    int  width, height, channels;
    auto image =
        stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    ASSERT_PANIC(image);
    Image_Raw out;
    out.width  = width;
    out.height = height;
    out.format = format;
    out.data.resize(width * height * 4u);
    memcpy(&out.data[0], image, out.data.size());
    stbi_image_free(image);
    return out;
  }
}

void save_image(std::string const &filename, Image_Raw const &image) {
  std::vector<u8> data;
  data.resize(image.width * image.height * 4);
  switch (image.format) {
  case vk::Format::eR32G32B32Sfloat: {

    ito(image.height) {
      jto(image.width) {
        vec3 *src = (vec3 *)&image.data[i * image.width * 12 + j * 12];
        u8 *  dst = &data[i * image.width * 4 + j * 4];
        vec3  val = *src;
        u8    r   = u8(255.0f * clamp(val.x, 0.0f, 1.0f));
        u8    g   = u8(255.0f * clamp(val.y, 0.0f, 1.0f));
        u8    b   = u8(255.0f * clamp(val.z, 0.0f, 1.0f));
        u8    a   = 255u;
        dst[0]    = r;
        dst[1]    = g;
        dst[2]    = b;
        dst[3]    = a;
      }
    }
  } break;
  default: ASSERT_PANIC(false && "Unsupported format");
  }
  stbi_write_png(filename.c_str(), image.width, image.height, STBI_rgb_alpha,
                 &data[0], image.width * 4);
}