import json

# Input and output file names
input_file = 'gen_A_data_reformatted.jsonl'
output_file = 'gen_A_final.jsonl'

def clean_and_rename(input_path, output_path):
    count = 0
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Check if 'prompt' field exists
                    if 'prompt' in data:
                        # Get the value
                        content = data['prompt']
                        
                        # Remove "Soru:\n" if it exists at the start
                        if content.startswith("Soru:\n"):
                            content = content.replace("Soru:\n", "", 1)
                        
                        # Create the new 'question' field with the cleaned content
                        data['question'] = content
                        
                        # Remove the old 'prompt' field
                        del data['prompt']
                    
                    # Write the modified object to file
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    count += 1
                    
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")
                    continue

        print(f"Processing complete.")
        print(f"Total records processed: {count}")
        print(f"Cleaned data saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    clean_and_rename(input_file, output_file)