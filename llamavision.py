import base64
import os
import argparse

import ollama

def filepath_to_base64(filepath):
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

SYSTEM_MESSAGES = [
    '''You are a Document information extraction OCR assistant for a university class that returns only well-formatted JSON.
The user inputs a single image and you reply with the following information extracted from the document:
- Name: The handwritten student's name
- UFID: The handwritten eight-digit id number 
- Section Number: The handwritten five-digit section number.
The ID is filled by the student in a row of adjacent cells. Its input label is "UFID". If you cannot find it, return an empty string.
The section number is filled by the student. The input label is "Section Number". If you cannot find it, return an empty string.
You do not say anything else in the response, just {"name": "val1", "ufid": "val2", "section_number": "val3"}, where val1 should be replaced handwritten student's name, where val2 should be replaced by the 8-digit ufid and val3 should be replaced by the 5-digit section number, as you see them in the document.''',
    '''You are a Document information extraction OCR assistant for a university class that returns only well-formatted JSON.
The user inputs a single image and you reply with the following information extracted from the document:
- "name": The handwritten student's name
- "ufid": The handwritten eight-digit id number
- "section_number": The handwritten five-digit section number.
- "problem_description": The description of the problem.
- "problem_number": The numeric label of the problem as it appears next to the problem description.
- "student_work": The full handwritten work of the student
- "student_final_answer": The final numeric answer at the end of the student's work.
You do not say anything else in the response, just the JSON.''',
]


def vision_chat_img(img, args):

    response = ollama.chat(
        stream=args['stream'],
        format='json',
        model=args['model'],
        messages=[
        {
            'role': 'system',
            'content': SYSTEM_MESSAGES[args['system']],
        },
        {
            'role': 'user',
            'images': [img]
        },
        ],
        options = {
            'top_k': args['top_k'],
        },
        )
    if args['stream']:
        message_chunks = []
        for chunk in response:
            type_msg = 'message' if 'message' in chunk else 'error'
            chuck_content = chunk[type_msg]['content']
            message_chunks.append(chuck_content)
            print(chuck_content, end='', flush=True)
        message = ''.join(message_chunks)
    else:
        message = response['message']['content']
        print(message)
    print()
    return message

def get_img_digit_probs(img, args):
    aggretated_responses = []
    for _ in range(args["n_trials"]):
        response = vision_chat_img(img, args)
        aggretated_responses.append(response)
    return get_digit_probs(aggretated_responses)

def vision_chat_digit_probs(args):
    filepath = args['filepath']
    if os.path.isfile(filepath) and filepath.endswith(".png"):
        img = filepath_to_base64(filepath)
        parse_result = get_img_digit_probs(img, args)
        return [parse_result,]
    # if is folder, get all PNG files
    if not os.path.isdir(filepath):
        print("Invalid path.")
        return None
    results = []
    for file in os.listdir(filepath):
        if not file.endswith(".png"):
            continue
        # if it doesnt include the pattern then skip
        if args['pattern'] not in file:
            continue
        img = filepath_to_base64(os.path.join(filepath, file))
        parse_result = get_img_digit_probs(img, args)
        results.append(parse_result)
    return results

def get_digit_probs(responses):
    '''Converts the response to a JSON object, extracts the field ufid, and returns the probabilities of each digit.'''
    import json
    ufids = []
    UFID_LENGTH = 8
    for response in responses:
        response = json.loads(response)
        ufid = str(response['ufid'])
        # if ufid is \d{8} then add to list
        if len(ufid) == UFID_LENGTH and ufid.isdigit():
            ufids.append(ufid)
    if not ufids:
        print("No valid UFID found.")
        return None
    probs = []
    for pos in range(UFID_LENGTH):
        digit_probs = [1] * 10
        for ufid in ufids:
            digit = int(ufid[pos])
            digit_probs[digit] += 1
        regularized_length = len(ufids) + 10
        digit_probs = [p / regularized_length for p in digit_probs]
        print(f"Position {pos}: {digit_probs}")
        probs.append(digit_probs)
    return probs

# example filepath: "imgs/sub-page-3.png"

if __name__ == '__main__':
    # example usage:
    # python llamavision.py --filepath imgs/ --includes "page-3" --n_trials 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='imgs/example.png')
    parser.add_argument('--pattern', type=str, default='')
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--system', type=int, default=0)
    parser.add_argument('--model', type=str, default='llama3.2-vision')
    parser.add_argument('--no-stream', action='store_false', dest='stream')
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    vision_chat_digit_probs(vars(args))
