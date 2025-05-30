import numpy as np
import gradio as gr
import torch
import os
import shutil
import gc

from PIL import Image, ImageDraw
from utils.sam_utils import show_masks
from utils.general import (
    get_coords, 
    plot_image, 
)
from utils.model_utils import (
    get_molmo_output, get_sam_output, get_spacy_output
)
from utils.load_models import (
    load_molmo, load_sam, load_siglip, load_spacy
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

molmo_model_name = None
sam_model_name = None
processor, molmo_model = None, None
sam_predictor = None

def process_image(
    image_path, 
    prompt,
    molmo_tag,
    sam_tag,
    clip_label,
    draw_bbox,
    sequential_processing,
    random_color,
    chat_only
):
    """
    Function combining all the components and returning the final 
    segmentation map.

    :param image: PIL image.
    :param prompt: User prompt.
    :param molmo_tag: Molmo Hugging Face model tag.
    :param sam_tag: SAM Hugging Face model tag.
    :param clip_label: Whether to enable auto labeling using CLIP.

    Returns:
        fig: Final segmentation map.
        prompt: Prompt from the Molmo model.
    """

    global molmo_model_name
    global sam_model_name
    global processor
    global molmo_model
    global sam_predictor
    global clicked_points

    coords = []

    image = Image.open(image_path)

    # Check if user chose different model, and load appropriately.
    if molmo_tag != molmo_model_name:
        gr.Info(message=f"Loading {molmo_tag}", duration=20)
        processor, molmo_model = load_molmo(model_name=molmo_tag, device=device)
        molmo_model_name = molmo_tag

    if not chat_only: # Load SAM only if `chat_only` mode is not selected.
        if sam_tag != sam_model_name:
            gr.Info(message=f"Loading {sam_tag}", duration=20)
            sam_predictor = load_sam(model_name=sam_tag)
            sam_model_name = sam_tag

    print(prompt)

    # Get coordinates from the model output.
    output = get_molmo_output(
        image, 
        processor,
        molmo_model,
        prompt
    )

    molmo_output = get_coords(output, image)

    if type(molmo_output) == str and len(clicked_points) == 0: # If we get image caption instead of points.
        # Clear mouse click prompts after one successful run.
        clicked_points = []
        return plot_image(image), output
    
     # There is a chance the user clicks points and Molmo outputs string. In
     # that case, we do not want to append the Molmo string output to `coords`
     # but only the clicked points.
    if type(molmo_output) != str:
        coords.extend(molmo_output)
    
    # Append the clicked points to `coords`.
    coords.extend(clicked_points)
    
    # Load CLIP and Spacy models if `clip_label` is True.
    if clip_label:
        spacy_nlp = load_spacy()
        # clip_processor, clip_model = load_clip()
        clip_processor, clip_model = load_siglip()

        # Get the nouns list.
        nouns = get_spacy_output(outputs=output, model=spacy_nlp)
    
    # Prepare input for SAM
    input_points = np.array(coords)
    input_labels = np.ones(len(input_points), dtype=np.int32)
    
    # Convert image to numpy array if it's not already.
    if isinstance(image, Image.Image):
        image = np.array(image)

    if chat_only:
        gr.Warning('Chat Only mode chosen. Ignoring every other option.')
    
    if not chat_only and clip_label: # If CLIP auto-labelling is enabled.
        label_array = [] # To store CLIP label after each loop.
        final_mask = np.zeros_like(image.transpose(2, 0, 1), dtype=np.float32)

        # This probably takes as many times longer as the number of objects
        # detected by Molmo.
        for input_point, input_label in zip(input_points, input_labels):
            masks, scores, logits, sorted_ind = get_sam_output(
                image,
                sam_predictor,
                input_points=[input_point],
                input_labels=[input_label]
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            final_mask += masks
        
            masks_copy = masks.copy()
            masks_copy = masks_copy.transpose(1, 2, 0)

            masked_image = (image * np.expand_dims(masks_copy[:, :, 0], axis=-1))
            masked_image = masked_image.astype(np.uint8)

            # Process masked image and give input to CLIP.
            clip_inputs = clip_processor(
                text=nouns, 
                images=Image.fromarray(masked_image), 
                return_tensors='pt', 
                padding=True
            )
            clip_outputs = clip_model(**clip_inputs)
            clip_logits_per_image = clip_outputs.logits_per_image # this is the image-text similarity score
            clip_probs = clip_logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
            clip_label = nouns[np.argmax(clip_probs.detach().cpu())]

            label_array.append(clip_label)

        im = final_mask >= 1
        final_mask[im] = 1
        final_mask[np.logical_not(im)] = 0
        
        fig = show_masks(
            image, 
            final_mask, 
            scores, 
            point_coords=input_points, 
            input_labels=input_labels, 
            borders=True,
            clip_label=label_array,
            draw_bbox=draw_bbox,
            random_color=random_color
        )

        # Clear mouse click prompts after one successful run.
        clicked_points = []
        return fig, output
    
    if not chat_only and sequential_processing: # If sequential processing of points is enabled without CLIP.
        final_mask = np.zeros_like(image.transpose(2, 0, 1), dtype=np.float32)

        # This probably takes as many times longer as the number of objects
        # detected by Molmo.
        for input_point, input_label in zip(input_points, input_labels):
            masks, scores, logits, sorted_ind = get_sam_output(
                image,
                sam_predictor,
                input_points=[input_point],
                input_labels=[input_label]
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            final_mask += masks
        
            masks_copy = masks.copy()
            masks_copy = masks_copy.transpose(1, 2, 0)

            masked_image = (image * np.expand_dims(masks_copy[:, :, 0], axis=-1))
            masked_image = masked_image.astype(np.uint8)

        im = final_mask >= 1
        final_mask[im] = 1
        final_mask[np.logical_not(im)] = 0
        
        fig = show_masks(
            image, 
            final_mask, 
            scores, 
            point_coords=input_points, 
            input_labels=input_labels, 
            borders=True,
            draw_bbox=draw_bbox,
            random_color=random_color
        )
        
        # Clear mouse click prompts after one successful run.
        clicked_points = []
        return fig, output
    
    else:
        masks, scores, logits, sorted_ind = None, None, None, None
        if not chat_only: # Get SAM output
            masks, scores, logits, sorted_ind = get_sam_output(
                image, sam_predictor, input_points, input_labels
            )
        
        # Visualize results.
        fig = show_masks(
            image, 
            masks, 
            scores, 
            point_coords=input_points, 
            input_labels=input_labels, 
            borders=True,
            draw_bbox=draw_bbox,
            chat_only=chat_only
        )
        
        # Clear mouse click prompts after one successful run.
        clicked_points = []
        return fig, output

# Global list to store clicked points
clicked_points = []

def draw_circle_on_img(img, center, radius=10, color=[255, 0, 0]):
    """Draw a circle on the image with the given radius and color."""
    x, y = center
    point_radius, point_color = 5, (255, 255, 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [
            (x - point_radius, y - point_radius), 
            (x + point_radius, y + point_radius)
        ],
        fill=point_color
    )
    return img

def get_click_coords(img, evt: gr.SelectData):
    """Handle the click event and draw a circle at the clicked coordinates."""
    global clicked_points

    out = Image.open(img)
    clicked_points.append((evt.index[0], evt.index[1]))
    
    for point in clicked_points:
        out = draw_circle_on_img(out, point)

    return out


with gr.Blocks(
    title='Image Segmentation with SAM2 and Molmo'
) as image_interface:
    # Inputs.
    img_input = gr.Image(type='filepath', label='Upload Image')
    txt_input = gr.Textbox(label='Prompt', placeholder='e.g., Point where the dog is.')
    
    # Outputs.
    img_plt_out = gr.Plot(label='Segmentation Result', format='png')
    molmo_out = gr.Textbox(label='Molmo Output')

    with gr.Row():
        with gr.Column():
            pointed_image = gr.Image(type='pil', label='Image with points')

    img_input.select(get_click_coords, [img_input], [pointed_image])
    
    # Additional inputs.
    molmo_models = gr.Dropdown(
        label='Molmo Models',
        choices=(
            'allenai/MolmoE-1B-0924',
            'allenai/Molmo-7B-O-0924',
            'allenai/Molmo-7B-D-0924',
            'allenai/Molmo-72B-0924',
        ),
        value='allenai/MolmoE-1B-0924'
    )

    sam_models = gr.Dropdown(
        label='SAM Models',
        choices=(
            'facebook/sam2.1-hiera-tiny',
            'facebook/sam2.1-hiera-small',
            'facebook/sam2.1-hiera-base-plus',
            'facebook/sam2.1-hiera-large',
        ),
        value='facebook/sam2.1-hiera-large'
    )

    clip_checkbox = gr.Checkbox(
        value=False, 
        label='Enable CLIP Auto Labelling.',
        info='Slower but gives better segmentations maps along with labels'
    )

    bbox_checkbox = gr.Checkbox(
        value=False, 
        label='Draw Bounding Boxes',
        info='Whether to draw bounding boxes around the segmentation objects. \
            Works best with CLIP Auto Labelling.'
    )

    seq_proc_checkbox = gr.Checkbox(
        value=False, 
        label='Sequential Processing',
        info='Process Molmo points sequentially generating one mask at a time. \
            Slower but more accurate masks.'
    )

    rnd_col_mask_checkbox = gr.Checkbox(
        value=False,
        label='Random color Mask',
        info='Randomly choose a mask color.'
    )

    chat_only_checkbox = gr.Checkbox(
        value=False,
        label='Chat Only Mode',
        info='If wanting pointing and chatting only without SAM \
            segmentation (saves inference time and memory). All other \
            above checkboxes will be ignored.'
    )

    gr.Interface(
        fn=process_image,
        inputs=[
            img_input, txt_input
        ],
        outputs=[
            img_plt_out, molmo_out
        ],
        additional_inputs=[
            molmo_models,
            sam_models,
            clip_checkbox,
            bbox_checkbox,
            seq_proc_checkbox,
            rnd_col_mask_checkbox,
            chat_only_checkbox
        ],
        description=f"Upload an image and provide a prompt to segment specific objects in the image."
    )

if __name__ == '__main__':
    # An output directory to save results.
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Gradio interface - now without video tab
    image_interface.launch(share=True)