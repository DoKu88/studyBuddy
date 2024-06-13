from ChatGPTInterface import ChatGPTInterface

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()
    

def main():

    fileText = read_file("./data/randomFile.cpp")

    #print(fileText)

    llmInterface = ChatGPTInterface()
    print(llmInterface.get_model_name())

    result = llmInterface.is_code_correct(fileText, "Does this use the leaky bucket algorithm?")
    resultDict = result.to_dict()
    print(resultDict['content'])

if __name__ == "__main__":
    main()