import json
import re

# Input and output file names
input_file = 'gen_A_final.jsonl'
output_file = 'gen_A_tokenized.jsonl'

def space_digits(text):
    """
    Inserts a space between every digit in a number found in the text.
    For example, "123" becomes "1 2 3".
    """
    if not text:
        return text
    # The regex (\d) matches any single digit.
    # We replace it with the digit followed by a space.
    # Then we might need to clean up extra spaces if necessary, 
    # but a simple approach is to find sequences of digits and space them out.
    
    def replacer(match):
        number = match.group(0)
        return " ".join(number)
    
    # Matches any sequence of digits
    return re.sub(r'\d+', replacer, text)

def process_line(data):
    # 1. Process the 'question' field
    if 'question' in data:
        data['question'] = space_digits(data['question'])
    
    # 2. Process the 'answer' field
    if 'answer' in data:
        content = data['answer']
        
        # Split the answer to isolate the "#### (number)" part
        # We look for the last occurrence of "\n####"
        separator = "\n####"
        parts = content.rsplit(separator, 1)
        
        if len(parts) == 2:
            main_text, final_number = parts
            
            # Apply spacing to the main text part
            main_text_spaced = space_digits(main_text)
            
            # Reassemble without spacing the final number part (the tag remains, the number remains compact)
            # The user said "besides the final \n#### (number) part it should remain as it is"
            # This implies the number AFTER #### should also NOT be spaced?
            # "it should remain as it is" usually means the whole block "\n#### 123" -> "\n#### 123"
            # If the user wanted "\n#### 1 2 3", they would likely say so. 
            # Assuming the final format is required for evaluation scripts which often expect "#### 123".
            
            data['answer'] = f"{main_text_spaced}{separator}{final_number}"
        else:
            # If the separator isn't found, just process the whole string
            data['answer'] = space_digits(content)
            
    return data

def tokenize_numbers(input_path, output_path):
    count = 0
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    processed_data = process_line(data)
                    outfile.write(json.dumps(processed_data, ensure_ascii=False) + '\n')
                    count += 1
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")
                    continue

        print(f"Processing complete.")
        print(f"Total records processed: {count}")
        print(f"Tokenized data saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    tokenize_numbers(input_file, output_file)