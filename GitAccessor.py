import git

# Path to the repository
def get_git_commit_info(repo_path, num_commits=10):

    print("num_commits: ", num_commits)
    # Open the repository
    repo = git.Repo(repo_path)

    # Initialize an empty list to store the commit information
    commits = []

    count1 = 0   
    # Iterate over the commits in the repository
    for commit in repo.iter_commits():
        # Access commit information
        commit_hash = commit.hexsha
        commit_message = commit.message
        commit_author = commit.author.name
        commit_date = commit.authored_datetime

        commits.append({
            "hash": commit_hash,
            "author": commit_author,
            "date": commit_date,
            "message": commit_message
        })

        count1 += 1
        if count1 >= num_commits:
            break

    return commits

