import os # python access to files/folders
import sys # python can read terminal arguments

from sentence_transformers import SentenceTransformer # library to convert text into vectors
import numpy as np

SUPPORT_EXTENSION = [".py", ".js", ".ts", ".tsx"] #read only these files, coding files, not other things

model = SentenceTransformer("all-MiniLM-L6-v2") # downloads pretrained embedding model

def scan_files(directory):
    
    chunks = [] # store all code we find

    for root, dirs, files in os.walk(directory): # os.walks(), walks through the folder recursively
        dirs[:] = [d for d in dirs if d not in ["venv", ".git", "__pycache__"]] # folders not to go in to

        for file in files: # loop through each file in current folder
            if any(file.endswith(ext) for ext in SUPPORT_EXTENSION): # make sure each file ends with the correct extension
                filepath = os.path.join(root, file) #build a path for the files

                try:
                    with open(filepath, "r", encoding="utf-8") as f: # open a file and read it = "r" = read mode
                        content = f.read() # opens files and reads the content inside it

                        chunks.append({ # stores the file in memory
                            "file": filepath,
                            "text": content
                        })

                except Exception as e: # if read is failed, program continues 
                    print(f"Could not read {filepath}: {e}")
    return chunks

def search(query, chunks, embeddings, top_k=3):
    # convert user query into vectors
    query_vector = model.encode([query])[0]

    #compare query vector to every file/chunk vector
    scores = np.dot(embeddings, query_vector) / (
        np.linalg.norm(embeddings, axis=1) * 
        np.linalg.norm(query_vector)
    )

    # get indexes of best search 
    best_indexes = np.argsort(scores)[::-1][:top_k]

    # return best matching files/chunks
    results = []

    for index in best_indexes:
        results.append({
            "score": scores[index],
            "file": chunks[index]["file"],
            "content":chunks[index]["text"]
        })
    return results


if __name__ == "__main__": # run if executed directly 
    if len(sys.argv) < 2:
        print("Usage: python index.py <folder_payh>")
        sys.exit(1)
        # if user forgets the folder path this it catches that
        # tells user how to run it and exit

    folder_path = sys.argv[1] #grabs folder path from terminal
    chunks = scan_files(folder_path) # calls scanner function to scan through files

    texts = [chunk["text"] for chunk in chunks] # extracts file content

    embeddings = model.encode(texts) # converts chunks into vectors

    print(f"\nGenerated {len(embeddings)} embeddings\n")

    if len (chunks) == 0:
        print("No supported file was found.")
        sys.exit(0)

    print(embeddings[0]) # prints first embedding vectors

    print(f"\nFound {len(chunks)} supported files:\n") # prints number of files found

    print('Ready!')

    while True:
        query = input("\nSearch your code: ")

        if query.lower() == "exit":
            break
            
        results = search(query, chunks, embeddings)

        for result in results:
            print("\n--- MATCH ---")
            print("File:", result["file"])
            print("Score:", result["score"])

            print(result["content"][:500])

    for chunk in chunks:
        print(chunk["file"]) # prints file path 