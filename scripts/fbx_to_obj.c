#include "ufbx.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void print_error(const ufbx_error *error)
{
    char buffer[2048];
    ufbx_format_error(buffer, sizeof(buffer), error);
    fprintf(stderr, "%s\n", buffer);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input.fbx output.obj\n", argv[0]);
        return 2;
    }

    ufbx_load_opts opts = { 0 };
    opts.target_axes = ufbx_axes_right_handed_y_up;
    opts.target_unit_meters = 1.0;
    opts.evaluate_skinning = true;
    opts.ignore_missing_external_files = true;

    ufbx_error error;
    ufbx_scene *scene = ufbx_load_file(argv[1], &opts, &error);
    if (!scene) {
        print_error(&error);
        return 1;
    }

    FILE *file = fopen(argv[2], "wb");
    if (!file) {
        perror("fopen");
        ufbx_free_scene(scene);
        return 1;
    }

    fprintf(file, "# Converted with ufbx from %s\n", argv[1]);
    size_t vertex_offset = 0;
    size_t total_vertices = 0;
    size_t total_triangles = 0;

    for (size_t mesh_ix = 0; mesh_ix < scene->meshes.count; mesh_ix++) {
        ufbx_mesh *mesh = scene->meshes.data[mesh_ix];
        if (!mesh->skinned_position.exists || mesh->num_faces == 0) continue;

        size_t tri_capacity = mesh->max_face_triangles * 3;
        uint32_t *tri_indices = (uint32_t *)malloc(sizeof(uint32_t) * tri_capacity);
        if (!tri_indices) {
            fprintf(stderr, "Out of memory\n");
            fclose(file);
            ufbx_free_scene(scene);
            return 1;
        }

        for (size_t instance_ix = 0; instance_ix < mesh->instances.count; instance_ix++) {
            ufbx_node *node = mesh->instances.data[instance_ix];
            if (!node->visible) continue;

            fprintf(file, "g %s\n", node->name.data ? node->name.data : "mesh");
            for (size_t vertex_ix = 0; vertex_ix < mesh->skinned_position.values.count; vertex_ix++) {
                ufbx_vec3 position = mesh->skinned_position.values.data[vertex_ix];
                if (mesh->skinned_is_local) {
                    position = ufbx_transform_position(&node->geometry_to_world, position);
                }
                fprintf(file, "v %.9g %.9g %.9g\n", position.x, position.y, position.z);
            }

            for (size_t face_ix = 0; face_ix < mesh->num_faces; face_ix++) {
                size_t num_triangles = ufbx_triangulate_face(
                    tri_indices, tri_capacity, mesh, mesh->faces.data[face_ix]);
                for (size_t tri_ix = 0; tri_ix < num_triangles; tri_ix++) {
                    fprintf(file, "f");
                    for (size_t corner = 0; corner < 3; corner++) {
                        uint32_t corner_ix = tri_indices[tri_ix * 3 + corner];
                        uint32_t vertex_ix = mesh->skinned_position.indices.data[corner_ix];
                        fprintf(file, " %zu", vertex_offset + (size_t)vertex_ix + 1);
                    }
                    fprintf(file, "\n");
                }
                total_triangles += num_triangles;
            }

            vertex_offset += mesh->skinned_position.values.count;
            total_vertices += mesh->skinned_position.values.count;
        }
        free(tri_indices);
    }

    fclose(file);
    ufbx_free_scene(scene);
    fprintf(stdout, "vertices=%zu triangles=%zu output=%s\n", total_vertices, total_triangles, argv[2]);
    return 0;
}
