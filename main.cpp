#define UTILS_TL_IMPL
#define UTILS_TL_TMP_SIZE 1 << 27
//#define UTILS_TL_IMPL_DEBUG
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
    this->data   = (u8 *)tl_alloc(size);
    memcpy(this->data, data, size);
  }
  void release() {
    if (data != NULL) tl_free(data);
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

static inline float3 safe_normalize(float3 v) {
  return v / (glm::length(v) + 1.0e-5f);
}

struct Vertex_Full {
  float3      position;
  float3      normal;
  float3      binormal;
  float3      tangent;
  float2      u0;
  float2      u1;
  float2      u2;
  float2      u3;
  Vertex_Full transform(float4x4 const &transform) {
    Vertex_Full out;
    float4x4    cmat = cofactor(transform);
    out.position     = float3(transform * float4(position, 1.0f));
    out.normal       = safe_normalize(float3(cmat * float4(normal, 0.0f)));
    out.tangent      = safe_normalize(float3(cmat * float4(tangent, 0.0f)));
    out.binormal     = safe_normalize(float3(cmat * float4(binormal, 0.0f)));
    out.u0           = u0;
    out.u1           = u1;
    out.u2           = u2;
    out.u3           = u3;
    return out;
  }
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
  u32                        num_vertices;
  u32                        num_indices;

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
        memcpy(&v.position,
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
      o.i0 = (u32) * (u16 *)index_data.at(2 * (id * 3 + 0));
      o.i1 = (u32) * (u16 *)index_data.at(2 * (id * 3 + 1));
      o.i2 = (u32) * (u16 *)index_data.at(2 * (id * 3 + 2));
    } else {
      o.i0 = (u32) * (u32 *)index_data.at(4 * (id * 3 + 0));
      o.i1 = (u32) * (u32 *)index_data.at(4 * (id * 3 + 1));
      o.i2 = (u32) * (u32 *)index_data.at(4 * (id * 3 + 2));
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
  Vertex_Full interpolate_vertex(u32 index, float2 uv) {
    Triangle_Full face = fetch_triangle(index);
    Vertex_Full   v0   = face.v0;
    Vertex_Full   v1   = face.v1;
    Vertex_Full   v2   = face.v2;
    float         k1   = uv.x;
    float         k2   = uv.y;
    float         k0   = 1.0f - uv.x - uv.y;
    Vertex_Full   vertex;
    vertex.normal =
        safe_normalize(v0.normal * k0 + v1.normal * k1 + v2.normal * k2);
    vertex.position = v0.position * k0 + v1.position * k1 + v2.position * k2;
    vertex.tangent =
        safe_normalize(v0.tangent * k0 + v1.tangent * k1 + v2.tangent * k2);
    vertex.binormal =
        safe_normalize(v0.binormal * k0 + v1.binormal * k1 + v2.binormal * k2);
    vertex.u0 = v0.u0 * k0 + v1.u0 * k1 + v2.u0 * k2;
    vertex.u1 = v0.u1 * k0 + v1.u1 * k1 + v2.u1 * k2;
    vertex.u2 = v0.u2 * k0 + v1.u2 * k1 + v2.u2 * k2;
    vertex.u3 = v0.u3 * k0 + v1.u3 * k1 + v2.u3 * k2;
    return vertex;
  }
};

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
    ito(images.size) images[i].release();
    images.release();
    ito(meshes.size) meshes[i].release();
    meshes.release();
    materials.release();
    ito(nodes.size) nodes[i].release();
    nodes.release();
  }
};

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
  static constexpr u32 MAX_ITEMS          = 4;         // max items
  // Node flags:
  static constexpr u32 FIRST_CHILD_MASK  = 0xffffff;
  static constexpr u32 FIRST_CHILD_SHIFT = 0;
  static constexpr u32 MAX_DEPTH         = 16;

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
    float3 orig = (min + max) / 2.0f;
    float3 size = (max - min) / 2.0f;
    float3 dr   = orig - ro;
    float3 invd = 1.0f / rd;
    float  dx_n = (dr.x - size.x) * invd.x;
    float  dy_n = (dr.y - size.y) * invd.y;
    float  dz_n = (dr.z - size.z) * invd.z;
    float  dx_f = (dr.x + size.x) * invd.x;
    float  dy_f = (dr.y + size.y) * invd.y;
    float  dz_f = (dr.z + size.z) * invd.z;
    float  nt   = MAX3(MIN(dx_n, dx_f), MIN(dy_n, dy_f), MIN(dz_n, dz_f));
    float  ft   = MIN3(MAX(dx_n, dx_f), MAX(dy_n, dy_f), MAX(dz_n, dz_f));
    return nt < ft;
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
    if (tris.size >= max_items && depth < BVH_Node::MAX_DEPTH) {
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
      node->init_leaf(hnode->min, hnode->max, alloc_tri_chunk());
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
    node_pool = Pool<BVH_Node>::create(1 << 16);
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

Image2D_Raw load_image(string_ref filename,
                       Format_t   format = Format_t::RGBA8_SRGB) {
  TMP_STORAGE_SCOPE;
  if (stref_find(filename, stref_s(".hdr")) != -1) {
    int            width, height, channels;
    unsigned char *result;
    FILE *         f = stbi__fopen(stref_to_tmp_cstr(filename), "rb");
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
    ASSERT_PANIC(hdr);
    Image2D_Raw out;
    out.init(width, height, Format_t::RGB32_FLOAT, (u8 *)hdr);
    stbi_image_free(hdr);
    return out;
  } else {
    int  width, height, channels;
    auto image = stbi_load(stref_to_tmp_cstr(filename), &width, &height,
                           &channels, STBI_rgb_alpha);
    ASSERT_PANIC(image);
    Image2D_Raw out;
    out.init(width, height, format, image);
    stbi_image_free(image);
    return out;
  }
}

void traverse_node(PBR_Model &out, aiNode *node, const aiScene *scene,
                   string_ref dir, u32 parent_id, float vk) {
  Transform_Node tnode;
  tnode.init();
  float4x4 transform;
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
    Raw_Mesh_Opaque opaque_mesh;
    opaque_mesh.init();
    using GLRF_Vertex_t      = Vertex_Full;
    opaque_mesh.num_vertices = mesh->mNumVertices;
    opaque_mesh.num_indices  = mesh->mNumFaces * 3;
    opaque_mesh.attribute_data.reserve(sizeof(GLRF_Vertex_t) *
                                       mesh->mNumVertices);
    opaque_mesh.index_data.reserve(sizeof(u32) * 3 * mesh->mNumFaces);
    auto write_bytes = [&](u8 *src, size_t size) {
      ito(size) opaque_mesh.attribute_data.push(src[i]);
    };
    auto write_index_data = [&](u8 *src, size_t size) {
      ito(size) opaque_mesh.index_data.push(src[i]);
    };
    bool has_textcoords    = mesh->HasTextureCoords(0);
    bool has_tangent_space = mesh->HasTangentsAndBitangents() && has_textcoords;
    ////////////////////////
    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
      GLRF_Vertex_t vertex;
      MEMZERO(vertex);
      vertex.position.x = mesh->mVertices[i].x * vk;
      vertex.position.y = mesh->mVertices[i].y * vk;
      vertex.position.z = mesh->mVertices[i].z * vk;
      if (has_tangent_space) {
        vertex.tangent.x = mesh->mTangents[i].x;
        vertex.tangent.y = mesh->mTangents[i].y;
        vertex.tangent.z = mesh->mTangents[i].z;

      } else {
        vertex.tangent = float3(0.0f, 0.0f, 0.0f);
      }
      if (has_tangent_space) {
        vertex.binormal.x = mesh->mBitangents[i].x;
        vertex.binormal.y = mesh->mBitangents[i].y;
        vertex.binormal.z = mesh->mBitangents[i].z;

      } else {
        vertex.binormal = float3(0.0f, 0.0f, 0.0f);
      }

      vertex.normal.x = mesh->mNormals[i].x;
      vertex.normal.y = mesh->mNormals[i].y;
      vertex.normal.z = mesh->mNormals[i].z;
      if (has_tangent_space) {
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
      }
      if (has_textcoords) {
        vertex.u0.x = mesh->mTextureCoords[0][i].x;
        vertex.u0.y = mesh->mTextureCoords[0][i].y;
      } else {
        vertex.u0 = glm::vec2(0.0f, 0.0f);
      }
      write_bytes((u8 *)&vertex, sizeof(vertex));
    }
    for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
      aiFace face = mesh->mFaces[i];
      for (unsigned int j = 0; j < face.mNumIndices; ++j) {
        write_index_data((u8 *)&face.mIndices[j], 4);
      }
    }
    opaque_mesh.index_type = Index_t::U32;

    aiMaterial * material = scene->mMaterials[mesh->mMaterialIndex];
    PBR_Material out_material;
    MEMZERO(out_material);
    float     metal_base;
    float     roughness_base;
    aiColor4D albedo_base;
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
    for (int tex = aiTextureType_NONE; tex <= aiTextureType_UNKNOWN; tex++) {
      aiTextureType type = static_cast<aiTextureType>(tex);

      if (material->GetTextureCount(type) > 0) {
        aiString relative_path;

        material->GetTexture(type, 0, &relative_path);
        char full_path[0x100];
        snprintf(full_path, sizeof(full_path), "%.*s/%s", STRF(dir),
                 relative_path.C_Str());
        Format_t format = Format_t::RGBA8_SRGB;

        switch (type) {
        case aiTextureType_NORMALS:
          format                 = Format_t::RGBA8_UNORM;
          out_material.normal_id = i32(out.images.size);
          break;
        case aiTextureType_DIFFUSE:
          out_material.albedo_id = i32(out.images.size);
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
          format              = Format_t::RGBA8_UNORM;
          out_material.arm_id = i32(out.images.size);
          break;
        default:
          fprintf(stderr, "[LOAD][WARNING] Unrecognized image type\n");
          //          ASSERT_PANIC(false && "Unsupported texture type");
          break;
        }
        out.images.push(load_image(stref_s(full_path), format));
      } else {
      }
    }
    opaque_mesh.vertex_stride = sizeof(GLRF_Vertex_t);
    // clang-format off
    opaque_mesh.attributes.push({Attribute_t::POSITION, Format_t::RGB32_FLOAT, OFFSETOF(Vertex_Full, position)});
    opaque_mesh.attributes.push({Attribute_t::NORMAL,   Format_t::RGB32_FLOAT, OFFSETOF(Vertex_Full, normal)});
    if (has_tangent_space) {
        opaque_mesh.attributes.push({Attribute_t::BINORMAL, Format_t::RGB32_FLOAT, OFFSETOF(Vertex_Full, binormal)});
        opaque_mesh.attributes.push({Attribute_t::TANGENT,  Format_t::RGB32_FLOAT, OFFSETOF(Vertex_Full, tangent)});
    }
    if (has_textcoords) {
        opaque_mesh.attributes.push({Attribute_t::UV0,      Format_t::RG32_FLOAT,  OFFSETOF(Vertex_Full, u0)});
        opaque_mesh.attributes.push({Attribute_t::UV1,      Format_t::RG32_FLOAT,  OFFSETOF(Vertex_Full, u1)});
        opaque_mesh.attributes.push({Attribute_t::UV2,      Format_t::RG32_FLOAT,  OFFSETOF(Vertex_Full, u2)});
        opaque_mesh.attributes.push({Attribute_t::UV3,      Format_t::RG32_FLOAT,  OFFSETOF(Vertex_Full, u3)});
    }
    out.materials.push(out_material);
    out.meshes.push(opaque_mesh);
    tnode.meshes.push(u32(out.meshes.size - 1));
  }
  out.nodes.push(tnode);
  out.nodes[parent_id].children.push(u32(out.nodes.size - 1));
  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    traverse_node(out, node->mChildren[i], scene, dir,
                  u32(out.nodes.size - 1), vk);
  }
}

PBR_Model load_gltf_pbr(string_ref filename) {
  Assimp::Importer importer;
  PBR_Model        out;
  out.init();
  out.nodes.push(Transform_Node{});
  string_ref dir_path = get_dir(filename);
  TMP_STORAGE_SCOPE;
  const aiScene *scene =
      importer.ReadFile(stref_to_tmp_cstr(filename),
                        aiProcess_Triangulate |              //
                            aiProcess_GenSmoothNormals |     //
                            aiProcess_PreTransformVertices | //
                            aiProcess_OptimizeMeshes |       //
                            aiProcess_CalcTangentSpace |     //
                            aiProcess_FlipUVs);
  if (!scene) {
    fprintf(stderr, "[FILE] Errors: %s\n", importer.GetErrorString());
    ASSERT_PANIC(false);
  }
  vec3 max = vec3(-1.0e10f);
  vec3 min = vec3(1.0e10f);
  calculate_dim(scene, scene->mRootNode, min, max);
  vec3 max_dims = max - min;

  // Size normalization hack
  float vk      = 1.0f;
  float max_dim = std::max(max_dims.x, std::max(max_dims.y, max_dims.z));
  vk            = 50.0f / max_dim;
  vec3 avg      = (max + min) / 2.0f;
  traverse_node(out, scene->mRootNode, scene, dir_path, 0, vk);

  out.nodes[0].offset = -avg * vk;
  return out;
}

struct Camera {
    float3 position;
    float3 look;
    float3 up;
    float3 right;
    float fov;
};
Camera gen_camera(
    float  phi,
    float  theta,
    float  r,
    float3 lookat,
    float  fov) {
    Camera cam;
    cam.position    =  r * float3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
    cam.look   =  -normalize(cam.position);
    cam.position    += lookat;
    cam.right  =  -normalize(cross(cam.look, float3(0.0, 1.0, 0.0)));
    cam.up     =  -cross(cam.right, cam.look);
    cam.fov    =  fov;
    return cam;
}
Ray gen_ray(Camera cam, float2 uv) {
    Ray r;
    r.o = cam.position;
    r.d = normalize(cam.look + cam.fov * (cam.right * uv.x + cam.up * uv.y));
    return r;
}
int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  //vec_test();
  //sort_test();
  PBR_Model model = load_gltf_pbr(stref_s("models/tree_low-poly_3d_model/scene.gltf"));
  int2 iResolution = int2(512, 512);
  float2 m = float2(0.0f, 0.0f);
  const float PI = 3.141592654f;
  Camera cam = gen_camera(
        0.0f,
        PI / 4.0f,
        60.0,
        float3(0.0, 0.0, 0.0),
        1.4
      );
  Array<BVH> bvhs;
  bvhs.init();
  defer({
      ito(bvhs.size) bvhs[i].release();    
      bvhs.release();
    });
  Pool<Tri> tri_pool = Pool<Tri>::create(1 << 20);
  defer(tri_pool.release());
  kto (model.meshes.size) {
     Raw_Mesh_Opaque &mesh = model.meshes[k];
     tri_pool.reset();
     Tri *tris = tri_pool.alloc(mesh.num_indices / 3);
     u32 num_tris = mesh.num_indices / 3;
     ito (num_tris) {
        Triangle_Full ftri = mesh.fetch_triangle(i);
        tris[i].a = ftri.v0.position;
        tris[i].b = ftri.v1.position;
        tris[i].c = ftri.v2.position;
        tris[i].id = i;
     }
     BVH bvh;
     bvh.init(tris, num_tris);
     bvhs.push(bvh);
    }
  {
      TMP_STORAGE_SCOPE;
      u8 *rgb_image = (u8*)tl_alloc_tmp(iResolution.x * iResolution.y * 3);
      ito(iResolution.y) {
        jto(iResolution.x) {
            float2 uv = float2((float(j) + 0.5f) / iResolution.y, (float(iResolution.y - i - 1) + 0.5f) / iResolution.y) * 2.0f - 1.0f;
            Ray ray = gen_ray(cam, uv);
            bool intersects = false;
            kto(model.meshes.size) {
                BVH &bvh = bvhs[k];
                bvh.traverse(ray.o, ray.d, [&](Tri &tri) {
                    Collision c;
                    if (ray_triangle_test_moller(ray.o, ray.d, tri.a, tri.b, tri.c, c))
                        intersects = true;    
                });

            }
            if (intersects) {
                rgb_image[i * iResolution.x * 3 + j * 3 + 0] = 255u;
                rgb_image[i * iResolution.x * 3 + j * 3 + 1] = 0u;
                rgb_image[i * iResolution.x * 3 + j * 3 + 2] = 0u;
            } else {
                rgb_image[i * iResolution.x * 3 + j * 3 + 0] = 0u;
                rgb_image[i * iResolution.x * 3 + j * 3 + 1] = 0u;
                rgb_image[i * iResolution.x * 3 + j * 3 + 2] = 0u;
            }
        }
      }
      write_image_2d_i24_ppm("image.ppm", rgb_image, iResolution.x * 3, iResolution.x, iResolution.y);
  }
  model.release();
  #ifdef UTILS_TL_IMPL_DEBUG
  assert_tl_alloc_zero();
  #endif
  return 0;
}