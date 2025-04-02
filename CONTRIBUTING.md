Contributing to CONTRIBUTING.md

## Table of Contents

- [Branching model](#branching-model)
- [Commit messages](#commit-messages)


## Branching model

All branches should start from the develop branch and be merged back to it through a Pull Request (PR)

-  **Master branch**: Changes to this branch are merged only through pull requests from the develop branch. This branch should be linked with the production environment
-  **Develop branch**: All branches start from this branch. Branching naming convention is the following:
- **feature/<feature-name>**: Used for specific feature work. Typically, this branches from and merges back into the development branch.
- **bugfix/<bugfix-name>**: Typically used for fixing bugs against a release branch
- **hotfix/<hotfix-name>**: Typically used to quickly fix the production branch


- Create a new branch and checkout to it

  ``` git checkout -b feature/feature-name```


- Fetch locally remote branches

  ```git fetch origin```

- Pull a branch
 
  ```git pull origin develop```

- Push changes to a branch

  ``` git push origin feature/feature-name```



## Commit messages
Inspired by:
  - https://www.conventionalcommits.org/en/v1.0.0/
  - https://github.com/RomuloOliveira/commit-messages-guide
  - https://cbea.ms/git-commit/
  - https://blog.devops.dev/how-to-validate-commit-messages-c67946eb469e

  Commit messages format:
```
<type>: <description>

[optional body]

[optional footer(s)]
```


types:
- **feat**: about a new feature implementation
- **fix**: about a bug fix
- **chore**: about changes that do not relate to a fix or a feature and do not modify src or test files (e.g., updating dependencies)
- **refactor**: about refactoring code that neither fixes a bug nor adds a feature
- **docs**: about updating documentation such as README and other markdown files or docstrings
- **style**: about changes that do not affect the meaning of the code but relate to code formatting
- **test**: about including new or correcting old tests
- **perf**: about performance improvements
- **ci**: about continuous integration
- **build**: about changes that affect the build system or external dependencies
- **revert**: reverts a previous commit

### Rules

- Capitalize the first letter:
  - to follow the grammar rule of using capital letters at the beginning of a sentence
- Do not end the subject with a period
  - trailing punctuation is unnecessary in subject lines
- Use a maximum of 50 characters for the subject
- Avoid generic messages or messages without any context
- Use imperative mood in the subject line
  - Git itself uses the imperative whenever is creates a commit on your behalf. Using the imperative, you follow git's build in conventions.
  - A properly formed Git commit message should always be able to complete the following sentence:
    - If applied, this commit will < <u>subject line</u> >
- Communicate what the change does without having to look at the source code
- The subject and the body are separated by a blank line
- Wrap the body at 72 characters
  - Git never wraps text automatically. You have to keep in mind its right margin and do it manually
- Use characters like - and * to improve readability
- Use the message body to explain "why", "for what", "how" and additional details
  - In most cases, you can leave out details about how a change has been made.
  - Just focus on making clear the reasons why you made the change.
- Add metadata in the footer such as the ticket number.
- Keep one blank line between body and footer

### Commit message example

```makefile
docs: Summarize changes in around 50 characters or less

More detailed explanatory text, if necessary. Wrap it to about 72
characters or so. In some contexts, the first line is treated as the
subject of the commit and the rest of the text as the body. The
blank line separating the summary from the body is critical (unless
you omit the body entirely); various tools like `log`, `shortlog`
and `rebase` can get confused if you run the two together.

Explain the problem that this commit is solving. Focus on why you
are making this change as opposed to how (the code explains that).
Are there side effects or other unintuitive consequences of this
change? Here's the place to explain them.

Further paragraphs come after blank lines.

 - Bullet points are okay, too

 - Typically a hyphen or asterisk is used for the bullet, preceded
   by a single space, with blank lines in between, but conventions
   vary here

If you use an issue tracker, put references to them at the bottom,
like this:

Resolves: #123
See also: #456, #789

```

### Git commit commands

- Use a signle commit message without body

  ``` git commit -m "Add only a single line message"```


- Opens a text editor to add a full commit message with with subject, body and footer

  ``` git commit```

  OR

  Use pycharm to write commits (cmd + K) or Git -> Commit

  OR

  Write your commit message in txt file and use ```git commit -F path/to/file``` to complete the commit


- Modify the most recent commit
  ```git commit --amend```:


- Modify the most recent commit by adding staged changes without changing the commit message

  ```git commit --amend --no-edit```


- Apply the changes from the specified commit onto the current branch
 
  ```git cherry-pick <commit>```


- Create a new commit that undoes the changes introduced by the specified commit

  ```git revert <commit>```


- Display detailed information about the specified commit, including the commit message, author, date, and the changes introduced.
 
  ```git show <commit>```

- Display a list of commits in reverse chronological order, showing the commit hash, author, date, and commit message.

  ```git log```


- Display a condensed list of commits with abbreviated commit hashes and commit messages on a single line.
 
  ```git log --oneline```


- Display commits by a specific author (replace "John" with the desired author name).

  ```git log --author="John"```


- Display commits that contain the specified keyword in the commit message.

  ```git log --grep="keyword"```


- Display the commit and author information for each line in a file.

  ```git blame <file> ```