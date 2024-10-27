import comfy.samplers
from typing import List
import random

import comfy.sample
import latent_preview
import torch
import comfy.utils


class Noise_EmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        return torch.zeros(
            latent_image.shape,
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = (
            input_latent["batch_index"] if "batch_index" in input_latent else None
        )
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)


class GridSwapper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The model used for denoising the input latent."},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "The number of steps used in the denoising process.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt.",
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."
                    },
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "tooltip": "The scheduler controls how noise is gradually removed to form the image."
                    },
                ),
                "positive": (
                    "CONDITIONING",
                    {
                        "tooltip": "The conditioning describing the attributes you want to include in the image."
                    },
                ),
                "negative": (
                    "CONDITIONING",
                    {
                        "tooltip": "The conditioning describing the attributes you want to exclude from the image."
                    },
                ),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.",
                    },
                ),
                "rows": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Number of grid rows.",
                    },
                ),
                "cols": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Number of grid columns.",
                    },
                )
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"  # TODO other category

    # not sure if this is really kendall tau distance
    def kendall_tau_distance(self, perm1: List[int], perm2: List[int]) -> int:
        """
        Calculate Kendall tau distance between permutations.
        Implementation optimized for speed using position lookup.
        """
        n = len(perm1)
        # Create position lookup for second permutation
        pos2 = {val: idx for idx, val in enumerate(perm2)}

        # Convert perm1 to relative positions in perm2
        relative_pos = [pos2[val] for val in perm1]

        # Count inversions using merge sort approach
        inversions = 0
        for i in range(n):
            for j in range(i + 1, n):
                if relative_pos[i] > relative_pos[j]:
                    inversions += 1
        return inversions

    def get_diverse_permutations_fast(
        self, n: int, k: int, num_candidates: int = 100
    ) -> List[List[int]]:
        """
        Get k diverse permutations using fast approximate method.

        Args:
            n: Length of array
            k: Number of permutations to return
            num_candidates: Number of random candidates to consider each iteration

        Returns:
            List of k diverse permutations
        """
        if k <= 0:
            return []

        # Start with first permutation
        result = [list(range(n))]

        # Helper function to generate random permutation
        def random_perm():
            perm = list(range(n))
            random.shuffle(perm)
            return perm

        # For each additional permutation needed
        for _ in range(k - 1):
            candidates = [random_perm() for _ in range(num_candidates)]

            # Find candidate with maximum minimum distance to existing permutations
            max_min_distance = -1
            best_candidate = None

            for candidate in candidates:
                # Calculate minimum distance to any existing permutation
                min_distance = min(
                    self.kendall_tau_distance(candidate, existing)
                    for existing in result
                )

                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate

            result.append(best_candidate)

        return result

    def combine_latents(self, samples, rows, cols):
        x = samples
        cell_count = rows * cols

        if x.shape[0] != cell_count:
            raise ValueError(f"Expected {cell_count} latent images, got {x.shape[0]}")

        dim = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        combined_h = h * rows
        combined_w = w * cols
        combined = torch.zeros(
            (1, dim, combined_h, combined_w), device=x.device, dtype=x.dtype
        )
        for i in range(rows):
            for j in range(cols):
                row_start = i * h
                row_end = row_start + h
                col_start = j * w
                col_end = col_start + w
                index = i * cols + j
                combined[0, :, row_start:row_end, col_start:col_end] = x[index]
        return combined

    def split_latents(self, combined, rows, cols):
        x = combined
        cell_count = rows * cols
        dim = x.shape[1]

        if x.shape[0] != 1:
            raise ValueError(f"Expected 1 latent image, got {x.shape[0]}")

        combined_h = x.shape[2]
        combined_w = x.shape[3]
        h = combined_h // rows
        w = combined_w // cols

        split = torch.zeros((cell_count, dim, h, w), device=x.device, dtype=x.dtype)
        for i in range(rows):
            for j in range(cols):
                index = i * cols + j
                row_start = i * h
                row_end = row_start + h
                col_start = j * w
                col_end = col_start + w
                split[index] = x[0, :, row_start:row_end, col_start:col_end]
        return split


    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
        rows=2,
        cols=2,
    ):
        cells = rows * cols

        latent = latent_image
        latent_image = latent["samples"]

        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
        latent["samples"] = latent_image

        latent_image = latent_image.clone()

        no_latents = latent_image.shape[0]

        if no_latents % cells != 0:
            raise ValueError(f"Number of latents ({no_latents}) is not a multiple of cells ({cells}), latents need to be divisible by cells. With {rows} rows x {cols} cols = {cells} cells, this means for example {1*cells}, {2*cells}, {3*cells}, ... latents are supported.")


        perms = self.get_diverse_permutations_fast(no_latents, steps)
        no_combined = latent_image.shape[0] // cells

        selected_latents = latent_image[range(cells)]
        samples_a = self.combine_latents(selected_latents, rows, cols)
        clatent = latent.copy()
        clatent["samples"] = samples_a

        noise = []
        for i in range(no_combined):
            noise.append(Noise_RandomNoise(seed + i).generate_noise(clatent))

        empty_noise = Noise_EmptyNoise().generate_noise(clatent)
        noise_mask = latent.get("noise_mask", None)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        callback = latent_preview.prepare_callback(model, steps)

        for i in range(steps):
            print(f"Step {i+1}/{steps}")
            p = perms[i]
            for j in range(0, len(p), cells):
                s_noise = noise[j // cells] if i == 0 else empty_noise
                selected_latents = latent_image[p[j : j + cells]]
                samples_a = self.combine_latents(selected_latents, rows, cols)

                start_step = i+1
                last_step = start_step + 1
                force_full_denoise = last_step == steps

                samples_a = comfy.sample.sample(
                    model,
                    s_noise,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    samples_a,
                    denoise=denoise,
                    disable_noise=(i > 0),
                    start_step=start_step,
                    last_step=last_step,
                    force_full_denoise=force_full_denoise,
                    noise_mask=noise_mask,
                    callback=callback,
                    disable_pbar=disable_pbar,
                    seed=seed,
                )

                split_a = self.split_latents(samples_a, rows, cols)
                latent_image[p[j : j + cells]] = split_a

        out = latent.copy()
        out["samples"] = latent_image
        return (out,)


NODE_CLASS_MAPPINGS = {
    "GridSwapper": GridSwapper,
}

NODE_DISPLAY_NAME_MAPPINGS = {}
