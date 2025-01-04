from openai import OpenAI

client = OpenAI()
import re

def read_file(file_path):
    """Reads the content of a .txt file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_into_chunks(content):
    """Splits the content into chunks based on subsections."""
    return re.split(r'\n\n+', content)  # Split by double newlines

def rewrite_chunk(chunk):
    if len(chunk.split()) < 20:
        return chunk

    try:
        response = client.chat.completions.create(model="gpt-4o",
        messages=[
            {"role": "system", "content": """
                    You are an assistant who writes in the style of Ernest Hemingway. Use impactful sentences and a simple, direct tone.
                    Don't use unnecessary metaphors.
                    Keep the in-text citations and references intact.
                    Do not add or remove any information.
                """},
            {"role": "user", "content": f"Rewrite this in Ernest Hemingway's tone:\n{chunk}"}
        ])
        print("processed chunk size: ", len(chunk.split()))
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error rewriting chunk: {e}")
        return chunk  # Return the original chunk if rewriting fails

def correct_size(text, words_goal):

    try:
        response = client.chat.completions.create(model="gpt-4o",
        messages=[
            {"role": "system", "content": """
                    You are an assistant who writes in the style of Ernest Hemingway. Use impactful sentences and a simple, direct tone.
                    Don't use unnecessary metaphors.
                    Keep the in-text citations and references intact.
                    Do not add or remove any information.
                """},
            {"role": "user", "content": f"Rewrite this in Ernest Hemingway's tone in maximum {words_goal} words:\n{text}"}
        ])
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error rewriting chunk: {e}")
        return text  # Return the original chunk if rewriting fails

def save_to_file(content, output_path):
    """Saves the final content to a .txt file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(content)

def main(input_file, output_file, words_goal):
    """Main function to process the file."""
    # Step 1: Read the input file
    content = read_file(input_file)
    print("input size: ", len(content.split()))

    # Step 2: Split content into chunks
    chunks = split_into_chunks(content)

    # Step 3: Rewrite each chunk
    rewritten_chunks = [rewrite_chunk(chunk) for chunk in chunks]

    # Step 4: Combine rewritten chunks
    final_content = '\n\n'.join(rewritten_chunks)
    print("output size: ", len(final_content.split()))

    if len(final_content.split()) > words_goal:
        print("correcting size")
        final_content = correct_size(final_content, words_goal)
        print("corrected size: ", len(final_content.split()))

    # Step 5: Save the final content to the output file
    save_to_file(final_content, output_file)

    print(f"Rewritten content saved to {output_file}")

if __name__ == "__main__":
    input_file = "ai-gen.txt"  # Replace with your input file path
    output_file = "humanized.txt"  # Replace with your desired output file path
    words_goal = 2200
    main(input_file, output_file, words_goal)
