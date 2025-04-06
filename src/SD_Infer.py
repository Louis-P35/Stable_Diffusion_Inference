import torch
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np

def generate_image_from_prompt_and_input(
    model_id: str, controlnet_model_id: str, input_image_path: str, prompt: str, negative_prompt: str, output_image_path: str,
    strength: float = 0.65, guidance_scale: float = 7.5, num_inference_steps: int = 50,
    controlnet_conditioning_scale: float = 1.0
) -> None:
    """
    Generates an image using Stable Diffusion based on an input image and a textual prompt.

    Parameters:
        model_id (str): Hugging Face model ID for Stable Diffusion model.
        controlnet_model_id (str): Hugging Face model ID for ControlNet model.
        input_image_path (str): Path to the input image file.
        prompt (str): Textual prompt guiding image generation.
        negative_prompt (str): Negative prompt to avoid unwanted features.
        output_image_path (str): Path to save the generated output image.
        strength (float): Degree to which the generation should follow the input image (0-1).
                          Higher values lead to more deviation.
        guidance_scale (float): How strongly the prompt will guide image generation.
        num_inference_steps (int): Number of inference steps for image generation.
        controlnet_conditioning_scale (float): Strength of ControlNet conditioning (0-1).

    Returns:
        None: The generated image is saved to the specified path.
    """

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the ControlNet model
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    # Load the Stable Diffusion image-to-image pipeline with ControlNet
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    # Load and preprocess the input image
    init_image = Image.open(input_image_path).convert("RGB")
    init_image = init_image.resize((768, 768))  # resizing to model's expected input dimensions

    # Avec ControlNet Tile, pas besoin de Canny : on passe directement l'image d'entrée
    control_image = init_image  # Tile utilise l'image brute comme condition

    # Generate Canny edge map from the input image
    #init_image_np = np.array(init_image)
    #canny_image = cv2.Canny(init_image_np, 100, 200)  # Adjust thresholds as needed
    # Save the Canny edge map for debugging (as a NumPy array)
    #cv2.imwrite("canny_edge_map.jpg", canny_image)
    #canny_image = canny_image[:, :, None]  # Add channel dimension
    #canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)  # Convert to 3-channel
    #canny_image = Image.fromarray(canny_image)  # Convert back to PIL Image

    # Run the pipeline to generate the image
    generated_images = pipe(
        prompt=prompt,
        image=init_image,
        control_image=control_image,  # Pass the Canny edge map as control input
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=controlnet_conditioning_scale
    ).images

    # Save the first generated image
    generated_images[0].save(output_image_path)


# Example usage
if __name__ == "__main__":
    MODEL_ID = "dreamlike-art/dreamlike-anime-1.0" #"runwayml/stable-diffusion-v1-5"  # You can replace this with any SD-based model from Hugging Face
    #CONTROLNET_MODEL_ID = "lllyasviel/sd-controlnet-canny"  # ControlNet Canny model
    CONTROLNET_MODEL_ID = "lllyasviel/control_v11f1e_sd15_tile"  # Modèle Tile
    INPUT_IMAGE_PATH = "../img/franck-pitiot-kaamelott-perceval.png"
    #INPUT_IMAGE_PATH = "../img/souriante-belle-jeune-femme-debout-posant_171337-11412.png"
    OUTPUT_IMAGE_PATH = "output.jpg"
    PROMPT = "photo anime, masterpiece, high quality, absurdres"
    NEGATIVE_PROMPT = "simple background, duplicate, retro style, low quality, lowest quality, 1980s, 1990s, 2000s, 2005 2006 2007 2008 2009 2010 2011 2012 2013, bad anatomy, bad proportions, extra digits, lowres, username, artist name, error, duplicate, watermark, signature, text, extra digit, fewer digits, worst quality, jpeg artifacts, blurry"

    generate_image_from_prompt_and_input(
        model_id=MODEL_ID,
        controlnet_model_id=CONTROLNET_MODEL_ID,
        input_image_path=INPUT_IMAGE_PATH,
        prompt=PROMPT,
        strength=0.5,  # Lower strength to preserve input
        guidance_scale=7.5,
        num_inference_steps=40,
        output_image_path=OUTPUT_IMAGE_PATH,
        negative_prompt=NEGATIVE_PROMPT,
        controlnet_conditioning_scale=1.0  # Adjust to control how strongly edges are enforced
    )

    #controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
    #pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    #    MODEL_ID, controlnet=controlnet, torch_dtype=torch.float16
    #)
    # Preprocess input image to generate a Canny edge map, then pass it to the pipeline