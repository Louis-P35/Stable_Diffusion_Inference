# Install the required packages using:
# pip install diffusers transformers torch torchvision accelerate Pillow

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

def generate_image_from_prompt_and_input(
    model_id: str, input_image_path: str, prompt: str, output_image_path: str,
    strength: float = 0.75, guidance_scale: float = 7.5, num_inference_steps: int = 50
) -> None:
    """
    Generates an image using Stable Diffusion based on an input image and a textual prompt.

    Parameters:
        model_id (str): Hugging Face model ID for Stable Diffusion model.
        input_image_path (str): Path to the input image file.
        prompt (str): Textual prompt guiding image generation.
        output_image_path (str): Path to save the generated output image.
        strength (float): Degree to which the generation should follow the input image (0-1).
                          Higher values lead to more deviation.
        guidance_scale (float): How strongly the prompt will guide image generation.
        num_inference_steps (int): Number of inference steps for image generation.

    Returns:
        None: The generated image is saved to the specified path.
    """

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the Stable Diffusion image-to-image pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    # Load and preprocess the input image
    init_image = Image.open(input_image_path).convert("RGB")
    init_image = init_image.resize((512, 512))  # resizing to model's expected input dimensions

    # Run the pipeline to generate the image
    generated_images = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images

    # Save the first generated image
    generated_images[0].save(output_image_path)


# Example usage
if __name__ == "__main__":
    MODEL_ID = "runwayml/stable-diffusion-v1-5"  # You can replace this with any SD-based model from Hugging Face
    INPUT_IMAGE_PATH = "../img/franck-pitiot-kaamelott-perceval.png"
    OUTPUT_IMAGE_PATH = "output.jpg"
    PROMPT = "Ghibli style"

    generate_image_from_prompt_and_input(
        model_id=MODEL_ID,
        input_image_path=INPUT_IMAGE_PATH,
        prompt=PROMPT,
        output_image_path=OUTPUT_IMAGE_PATH
    )