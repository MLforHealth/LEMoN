def vqg_batched(pipeline, captions, batch_size = 64, clf = False):

    if clf: 
        user_prompt_template = '''Please generate some visual questions to ask a multimodal large language model to identify if the caption of an image is correct. 
These questions will help determine if the image corresponds to the given caption. 
Remember that the goal is to ask visual questions that would lead to a "yes" answer if the label is correct.
For example, if the caption is "A boy in red shirt playing ball", the possible questions could be:
Is there a boy in the picture?
Is the boy wearing a red shirt?
Is the ball clearly visible in the scene?
Is the boy interacting with the ball, such as kicking, throwing, or holding it?

You should generate 6 most insightful questions, separated by new lines.
The caption is "%s".
'''
    else:
        user_prompt_template = '''Please generate some visual questions to ask a multimodal large language model to identify if the label of an image is correct. 
These questions will help determine if the object in the image corresponds to the given label.
Remember that the goal is to ask questions that would lead to a ‘yes’ answer if the label is correct
For example, if the label is "airplane", the possible questions could be: 
Does the image contain an airplane?
Is there an airplane in the image?
Can the object in the image be used to fly in the air?
Does the object in the image have wings?

You should generate 6 most insightful questions, separated by new lines.
The label is "%s".
'''
    
    base_prompt_template =  '''
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    '''

    prompts = [
        base_prompt_template.format(user_prompt=user_prompt_template % caption)
        for caption in captions
    ]

    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        curr_text_out = pipeline(batch_prompts, do_sample=False, max_new_tokens = 512)
        # Extract responses
        for output, prompt in zip(curr_text_out, batch_prompts):
            response = output[0]['generated_text'].split(prompt)[-1].strip()
            results.append(list(map(lambda x: x.strip(), filter(lambda x: len(x.strip()) > 0, response.lower().replace('| hidden answer: yes |', '').split('\n'))))[:6])
            
    return results