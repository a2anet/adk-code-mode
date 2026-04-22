# adk-code-mode

Add high-level coding guidelines and review notes to this file. Specific actionable items from code reviews should be captured here for the Python implementation.

## Workflow

- Check with the user before creating a commit.
- When proposing or writing commit messages, use backticks around code objects and use their full names, for example `A2ASession`.
- Commit messages should be specific about the actual high-level changes, not generic summaries. For example `chore: add git hooks, fix build errors, and update README.md`, instead of centering the subject line on a lower-level implementation detail.
- Prefer `README.md`, workflow files, and similar repo objects in commit subjects when those files are a meaningful part of the change.
- Include a detailed bullet list in the commit body covering the main changes, including important related changes not explicitly discussed in chat.

## Engineering Guidelines

- Constants and utility functions should be kept in a constants or utility file if used by more than one class. Otherwise they should be in the class.
