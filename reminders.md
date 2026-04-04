## Reminder for start-up usage:
Notes to start a virtual environment and update it.

1. If not created, run `python -m venv .venv` to create a .venv directory as a virtual environment.
2. Run `source .venv/bin/activate` to activate the virutal environment.
    1. Alternatively, some files (in Windows) is saved as 'source .venv/Scripts/activate' instead.
3. Run `pip install -r requirements.txt` to install required packages.
    1. Or run `pip freeze > requirements.txt` to generate new requirements.
4. Check if your code is up-to-date with the main branch via `git status`.
    1. If not up-to-date, use `git pull` to update your copy of the code with the most up-to-date one.


## How to check for updates:
**Remember that your virtual environment is not being updated real-time, so you need to do one of the following to make sure everything is up-to-date**

If you are unsure if something has been updated while you are working, use the following:
1. Run `git fetch` to fetch the status of changes
2. If happy that changes do not overlap with yours, use `git merge` to merge your local repository with the GitHub repository.

If you know that changes were made already and not interferring with your work, just use `git pull`.

## How to deploy/push code:
Every time you make a change, GitHub will create a copy of the brach for your changes to be made, please ensure that you deploy any relevant changes when needed for sharing.
1. To add a file, do `git add <file name>` and if you want to push all of your repository just replace filename with `.`
2. Type `git status` to check which changes will be committed.
3. To stage the file for deployment, type `git commit -m <comment>` with relevant comment
    1. If you want to remove files from staging area, type `git reset <filename>`.
4. Once satisfied, run `git push` to push all changes to the main branch

## How to use branches:
How to check branches.
1. Run `git branch` to see all existing branches
    1. Run `git branch -d <branch name>` to delete a branch, note you may not be able to restore deleted branch once pushed
2. If you are in the wrong branch, to switch branches `git switch <branch name>`.

## Notes:
1. To check your working environment, use `which python` to check.
2. Check `git --help` if confused.
3. To check logs of changes committed, use `git log`.
    1. If you want more information on a log change, copy the ID and add to `git show <id>` to see changes made
4. `git diff <branch/id 1> <branch/id 2>` can be used to check differences between branches or commit logs, leave variables blank (<>) to check difference between stage repository and currently edited repository.

# Naming Conventions
Follow PEP 8 style guide:
- *snake_case* for functions / objects / instance/ files
- *CapWords* for classes