import GitAccessor
from CodeBERT import CodeBERT
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import PLBartTokenizer, PLBartForConditionalGeneration

def main():
    # Your code here
    print("Hello, world!")

    # Path to the repository
    repo_path = "/Users/chad/Projects/coinTrackCase"

    '''
    # Get the commit information
    commits = GitAccessor.get_git_commit_info(repo_path, num_commits=1)
    print('hola')

    # Print the commit information
    for commit in commits:
        print(f"Commit: {commit['hash']}")
        print(f"Author: {commit['author']}")
        print(f"Date: {commit['date']}")
        print(f"Message: {commit['message']}")
        print()
        '''

    codeSnippet = """
        class BinarySearchTree:
            class Node:
                def __init__(self, key):
                    self.left = None
                    self.right = None
                    self.val = key

            def __init__(self):
                self.root = None

            def insert(self, key):
                if self.root is None:
                    self.root = self.Node(key)
                else:
                    self._insert(self.root, key)

            def _insert(self, root, key):
                if root is None:
                    return self.Node(key)
                else:
                    if root.val < key:
                        root.right = self._insert(root.right, key)
                    else:
                        root.left = self._insert(root.left, key)
                return root

            def inorder_traversal(self, root):
                res = []
                if root:
                    res = self.inorder_traversal(root.left)
                    res.append(root.val)
                    res = res + self.inorder_traversal(root.right)
                return res

            def search(self, key):
                return self._search(self.root, key)

            def _search(self, root, key):
                if root is None or root.val == key:
                    return root
                if root.val < key:
                    return self._search(root.right, key)
                return self._search(root.left, key)

        # Usage example
        bst = BinarySearchTree()
        bst.insert(10)
        bst.insert(20)
        bst.insert(5)
        print(bst.inorder_traversal(bst.root))  # Output: [5, 10, 20]
        print(bst.search(10))  # Output: <__main__.BinarySearchTree.Node object at ...>
        """

    # Initialize the CodeBERT model
    codebert1 = CodeBERT(model_name = "uclanlp/plbart-base", 
                         tokenizer = PLBartTokenizer, 
                         model = PLBartForConditionalGeneration)
    print(codebert1.get_model_name())

    model = codebert1.get_model()
    tokenizer = codebert1.get_tokenizer()
    print("Model ", model.config)
    print("Tokenizer ", tokenizer.vocab_size)

    inputs = tokenizer(codeSnippet, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    # Generate the summary
    summary_ids = model.generate(inputs['input_ids'], max_length=50, num_beams=4, early_stopping=True, no_repeat_ngram_size=2)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("Code Summary:", summary)

    #inputs = codebert1.tokenize(codeSnippet)
    #outputs = codebert1.generateSummary(inputs)
    #print("The summary is: ")
    #print(outputs)

if __name__ == "__main__":
    main()