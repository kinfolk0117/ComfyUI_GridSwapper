# Gridswapper
Gridswapper takes a batch of latents and spreads them over the necessary amount of grids. It then automatically shuffles the images in the grids for each step.
So, a batch of 12 latents for a 2x2 grid will generate 3 grid images in each step. It will then shuffle around the images for the next step. This makes it possible for all images to influence the others during the denoising process.
This approach works well for generating 2-4 grids. 


To improve convergence, especially when generating many grids, consider:

* Increasing the number of steps, (for example 4*batch size) - this makes it more likely that each image can influence all others during the shuffling process
* Using ancestral samplers (like euler_a) - the added noise at each step seems to help with consistency
* Train a lora on 2x2 grids of the type of pictures you want to generate.

---

<img width="1001" alt="Screenshot 2024-10-26 at 20 57 21" src="https://github.com/user-attachments/assets/901e9a62-0c06-420b-afa5-6701a784e858">
