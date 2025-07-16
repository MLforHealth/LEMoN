import time
import torch
import pandas as pd
import urllib3
import transformers

from lib.vdc.utils import parse_entity

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

number = {
    "0": "zero none nothing no",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}


def replace_numbers(text):
    tokens = text.split()
    for i in range(len(tokens)):
        if tokens[i] in number.keys():
            tokens[i] = number[tokens[i]]
    return " ".join(tokens)


def eval_common_qa(response, answer, pipeline):
    output = eval_llama(response, answer, pipeline)
    if "yes" in output:
        is_match = True
    else:
        is_match = False

    return output, is_match


def eval_class_specific_qa(response, answer):
    response = response.lower()
    answer = answer.lower()
    response = replace_numbers(response)
    answer = replace_numbers(answer)
    answer_tokens = answer.split()
    for answer_token in answer_tokens:
        if answer_token in response:
            return True
    return False


def classification_acc(syn_list, pred_text):
    words = parse_entity(pred_text)
    for syn in syn_list:
        if syn in words:
            return True
    return False

def get_vdc_ae_score(caption, general_responses, class_specific_responses, class_specific_answers, llm_pipeline):
    num_general = 0
    num_specific = 0
    num_general_match_true = 0
    num_specific_match_true = 0

    for i in range(len(general_responses)):
        is_match = eval_image(
        answer=caption,
        response=general_responses[i],
        question_type='common',
        pipeline = llm_pipeline)
        num_general_match_true+=int(is_match)
        num_general+=1

    for i in range(len(class_specific_answers)):
        is_match = eval_image(
                answer=class_specific_answers[i],
                response=class_specific_responses[i],
                question_type='uncommon')
        num_specific_match_true+=int(is_match)
        num_specific+=1

    return 1 - (num_general_match_true+num_specific_match_true)/(num_general+num_specific)


def eval_image(
    answer,
    response,
    question_type='common',
    pipeline = None
):
    # columns in df: img_id, question, response, target_label
    # common questions
    if question_type =='common': 
        llm_output, is_match = eval_common_qa(response=response, answer=answer, pipeline = pipeline)
        return is_match
    # class-specific questions
    else:
        return eval_class_specific_qa(response=response, answer=answer)


def eval_llama(response, answer, pipeline):    
    user_prompt = f"""Assume you are a helpful and precise assistant for evaluation. Please judge whether the 'Caption' of an image and one of the 'Labels' refer to the same object. Answer with yes or no.
    - Caption: [{response}]
    - Labels: [{answer}]"""
    
    base_prompt =  '''
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    '''

    prompt = base_prompt.format(user_prompt = user_prompt)
    curr_text_out = pipeline(prompt, do_sample = False,
                                 max_new_tokens=2)
    response = curr_text_out[0]['generated_text'].split(prompt)[-1].strip()

    return response.lower()
