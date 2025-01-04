# Setting Up the Environment

1. **Create a Virtual Environment**:

    ```bash
    python3 -m venv new_env
    ```

2. **Activate the Virtual Environment**:

    - On **macOS/Linux**:
        ```bash
        source new_env/bin/activate
        ```
    - On **Windows**:
        ```bash
        new_env\Scripts\activate
        ```

3. **Install Dependencies**:
    ```bash
    pip3 install -r requirements.txt
    ```

4. **Add OpenAI API Key**:
    ```bash
    export OPENAI_API_KEY=<your-api-key>
    ```
5. **Create a folder call pdfs**:
    ```bash
    mkdir pdfs
    ```
6. **Download all the reference PDFs that you want use as source for your critical review and save them in the ./pdfs folder**
7. **Run knowledge generation**:
    This code is slow and inefficient. takes a life time to go through the whole thing. Have a problem with it? make it faster yourself :)
    took around 40min to parse through 20 pdfs which was 27mb of information, so go watch a movie or something.
    Go inside the code (yes I know about environment varibles, while at it maybe you can do it) and config `batch_size`. I found out that having 5 to 10 articles in one batch is good. so for example if you have 21 articles have batches of 7. More articles in one batch might lead to token limitation and less feels too low. 
    ```bash
    python3 pdf2knowledge.py  
    ```
8. **Run article generation**:
    Go inside the code (again, I know. need it?, add it yourself) and make sure to update the following:
    - The task should be updated based on your research theme and subjects related to the theme that must be critiqued.
    - Replace the current list of refereneces with the list of references that you have and downloaded and saved in the ./pdfs 
    - Update `word_goal` and `total_repeats`. also make sure to update word count in the task text both in `main` function and `iterative_review_generation` function
    ```bash
    python generate-article.py
    ```


# Things to keep in mind
- Sometimes LLM goes crazy just rerun it
- Sometimes the final output has less cites that was originally given, this means that the cite you added might not be related to the others or didn't quite fit in with the whole thing. I had 21 cites but LLM always used 20 because one was not that relevant.
- Need more text? add more cites 