#include "rt.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include <assimp/contrib/stb_image/stb_image.h>
#include <assimp/pbrmaterial.h>


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
                       Format_t   format) {
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