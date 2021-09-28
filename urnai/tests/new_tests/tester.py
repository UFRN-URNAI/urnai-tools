import sys, pathlib, os
import subprocess
import getpass


class Errors:
    class GitHashFileNotFoundError(Exception):
        pass
    class UrnaiRepoNotFoundError(Exception):
        pass

def get_urnai_latest_commit_hash():
    sep = os.path.sep
    git_path = find_urnai_git_repo_dir()
    if git_path != None:
        print(git_path)
        latest_commit_hash = f"{gitpath}{sep}refs{sep}heads{sep}master"
        if os.exists(latest_commit_hash):
            with open(latest_commit_hash, 'r') as hash_:
                return hash_.read()
        else:
            raise Errors.GitHashFileNotFoundError(f"{latest_commit_hash}: file not found!")
    else:
        raise Errors.UrnaiRepoNotFoundError(f"Urnai repo: file not found!")

def get_pip_freeze():
    try:
        return subprocess.check_output("pip freeze")
    except subprocess.CalledProcessError as cpe:
        return cpe.stderr

def get_username():
    return getpass.getuser()

def find_urnai_git_repo_dir():
    sep = os.path.sep
    git_path = f"{str(pathlib.Path(__file__).parent.parent.parent.parent.parent)}{sep}.git"
    
    if os.exists(git_path):
        return git_path
    else:
        answer = get_string_input("I could not find urnai git repo, do you want me to search recursively on your home folder? [y/n]", "n").lower()
        if answer == "y":
            home_folder = os.expanduser('~')
            urnai_repos = find_dir("urnai-tools", home_folder)
            if len(urnai_repos) > 0:
                if len(urnai_repos) == 1:
                    if is_dir_there(".git", urnai_repos[0]):
                        return urnai_repos[0]
                    else:
                        raise FileNotFoundError(f"{urnai_repos[0]} does not contain a .git folder.")
                else:
                    print("There were multiple urnai repos in your home folder:")
                    for i in range(len(urnai_repos)): print(f"{i + 1} - {urnai_repos[i]}")
                    answer = get_int_input("\nWhich one do you want?", 1)
                    return urnai_repos[answer - 1]
            else:
                raise Errors.UrnaiRepoNotFoundError(f"Urnai repo: file not found!")
        else:
            return None

def is_dir_there(dir_name, where_to_find):
    return os.path.exists(f"{where_to_find}{os.path.sep}{dir_name}")

def find_dir(dir_name, where_to_find):
    dir_list = []
    for root, dirs, files in os.walk(where_to_find):
        if dir_name in dirs:
            dir_list.append(os.path.join(root, dir_name))

    if len(dir_list) == 0:
        raise FileNotFoundError(f"File {where_to_find}{os.path.sep}{dir_name} was not found!")
    else:
        return dir_list

def generate_log(user, pip_freeze, output, err):
    with open("geral_test_info.log",'w') as file_:
        text = """User: {usr}
        Packages installed (pip freeze): {pip_fr}""".format(usr=user, pip_fr=pip_freeze)
        file_.write(text)

    with open("test_output.log",'w' ) as file_:
        file_.write(str(output,'UTF-8'))

    with open("test_err.log",'w') as file_:
        file_.write(str(err,'latin-1'))

def main():
    #TODO generate a log of the test, plus username, urnai revision hash and pip freeze
    pip_freeze = get_pip_freeze()
    user = get_username()

    output, err = "", ""
    p = subprocess.Popen(["python3", "-m", "pytest", "test_cartpole.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()

    generate_log(user, pip_freeze, output, err)

    #handle if process returned error
    #if p.returncode != 0: 
    #       print()




#aux functions
def get_int_input(message, default_value):
    try:
        return int(input(message + " [Default " + str(default_value) + "] "))
    except ValueError:
        return default_value

def get_string_input(message, default_value):
    try:
        string = str(input(message + " [Default " + str(default_value) + "] "))
        return string if string != "" else default_value
    except ValueError:
        return default_value

#main
main()
