"""
This is a Python application that utilizes OCR (Optical Character Recognition) and GPT (Generative Pre-trained Transformer) to process and analyze images containing text.
The application extracts text from the image, corrects any OCR errors using GPT, and provides an opinion or judgment on the image's information.
The user can interact with the application through a Gradio user interface,
which allows them to input an image, select a GPT model, and adjust the temperature setting for the GPT response.
url: https://github.com/NowLoadY
"""
import easyocr
import gradio as gr
import openai
import webbrowser
import re
from spellchecker import SpellChecker
import pyperclip

openai.api_key = ""

# Initialize OCR reader
reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)

# Define the text preprocessing function
def preprocess_ocr_result(ocr_result):
    # Remove empty strings and special characters
    cleaned_result = [re.sub(r'[^\w！，。、,.]+', '', word) for word in ocr_result if word.strip()]
    print("cleaned_result:{}".format(cleaned_result))
    # Correct spelling errors
    # spell = SpellChecker()
    # cleaned_result1 = [spell.correction(word) for word in cleaned_result]
    # print("cleaned_result1:{}".format(cleaned_result1))
    return cleaned_result

# Define the OCR function
def ocr_gpt(image, api_key, model, temperature):
    openai.api_key = api_key
    ocr_results_with_coordinates = reader.readtext(image)

    def distance(point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def compare_results(result1, result2):
        point1 = result1[0][0]
        point2 = result2[0][0]
        return distance(point1, point2)

    sorted_ocr_results = sorted(ocr_results_with_coordinates, key=lambda x: (x[0][0][1], x[0][0][0]))
    sorted_ocr_results = sorted(sorted_ocr_results, key=lambda x: compare_results(x, sorted_ocr_results[0]))
    ocr_result = [result[1] for result in sorted_ocr_results]
    cleaned_ocr_result = preprocess_ocr_result(ocr_result)
    print("ocr_result: ", ocr_result)
    prompt = "Please correct the following OCR results: " + str(cleaned_ocr_result) + ", you can remove any words you think are meaningless and reply with the corrected results in the form of a Python list"
    print(prompt)
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens":250,
        "n": 1,
        "stop": None,
        "temperature": temperature,
    }
    response = openai.Completion.create(**data)
    corrected_text = response.choices[0].text.strip().replace(".", "").replace("\n", "").replace(":", "").replace(" ", "")
    print("corrected_text: " + corrected_text)
    gpt_opinion_prompt = "The OCR result is: {}. Please provide your opinion or make a judgment, and ensure your response does not contain any line breaks.".format(corrected_text)
    chat_data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": f"You are a helpful assistant, job is to give opinions"},
            {"role": "user", "content": gpt_opinion_prompt}
        ],
        "max_tokens": 100,
        "n": 1,
        "stop": None,
        "temperature": 0.5,
    }
    gpt_opinion_response = openai.ChatCompletion.create(**chat_data)
    gpt_opinion = gpt_opinion_response.choices[0].message['content'].strip()
    print("gpt_opinion_response:"+gpt_opinion)
    return str(ocr_result), corrected_text, gpt_opinion

def copy_to_clipboard(text):
    pyperclip.copy(text)
# Define Gradio UI components
themes = ['freddyaboulton/test-blue', 'gstaff/xkcd', 'gstaff/whiteboard', 'ParityError/Anime']

def update_theme(theme):
    start(theme=theme)
    
def start(theme=themes[2]):
    with gr.Blocks(theme=theme) as app:
        with gr.Tab("OCR-GPT"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.inputs.Image()
                    api_key_input = gr.inputs.Textbox(lines=1, label="API Key", default="")
                    model_input = gr.inputs.Dropdown(choices=["text-davinci-003", "text-davinci-002", "text-davinci-001"], label="GPT Model")
                    temperature_input = gr.inputs.Slider(minimum=0.1, maximum=1.0, step=0.1, default=0.5, label="Temperature")
                    submit_button = gr.Button("submit")
                    theme_selector = gr.inputs.Dropdown(choices=themes, label="Select Theme", default=themes[2])
                    
                with gr.Column():
                    original_ocr_text = gr.outputs.Textbox(label="OCR Result original")
                    corrected_ocr_output = gr.outputs.Textbox(label="OCR Result (GPT Corrected)")
                    gpt_opinion_output = gr.outputs.Textbox(label="GPT Opinion on Image Information")
                    copy_original_button = gr.Button("Copy Original OCR")
                    copy_corrected_button = gr.Button("Copy Corrected OCR")
            submit_button.click(ocr_gpt, inputs=[image_input, api_key_input, model_input, temperature_input], outputs=[original_ocr_text, corrected_ocr_output, gpt_opinion_output])
            copy_original_button.click(copy_to_clipboard, inputs=[original_ocr_text])
            copy_corrected_button.click(copy_to_clipboard, inputs=[corrected_ocr_output])
            theme_selector.change(update_theme, inputs=[theme_selector])
            
    app.launch(share=True)
    webbrowser.open('http://127.0.0.1:7860')
if __name__ == "__main__":
    start()