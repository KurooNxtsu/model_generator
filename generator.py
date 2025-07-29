import torch
import os
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.image_util import load_image
from shap_e.util.notebooks import decode_latent_mesh
from IPython.display import display
output_dir=r"C:\Users\nitis\Downloads\Model_mesh_generator\Output"
os.makedirs(output_dir,exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
input_type = input("Enter type of 3D model generation - 'text' or 'image': ").strip().lower()
batch_size = 4
render_mode = 'nerf'  
size = 64 
cameras = create_pan_cameras(size, device)
if input_type == 'text':
    prompt = input("Enter the text description for the 3D model: ").strip()
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    guidance_scale = 15.0
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

elif input_type == 'image':
    image_path = input("Enter the image path or URL: ").strip()

    if not os.path.exists(image_path):
        print("Error: File does not exist at provided path.")
        exit(1)

    image = load_image(image_path)

    model = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    guidance_scale = 3.0
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(images=[image] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
else:
    print("Invalid input type. Please choose either 'text' or 'image'.")
    exit(1)

for i, latent in enumerate(latents):
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    display(gif_widget(images))
    mesh = decode_latent_mesh(xm, latent).tri_mesh()
    ply_path = os.path.join(output_dir, f'output_model_{i}.ply')
    obj_path = os.path.join(output_dir, f'output_model_{i}.obj')

    with open(ply_path, 'wb') as f:
        mesh.write_ply(f)

    with open(obj_path, 'w') as f:
        mesh.write_obj(f)

print("3D model generation complete and saved as .ply and .obj files.")



