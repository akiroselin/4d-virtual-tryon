# Bash read-only security model

Easy Agent automatically approves a Bash command as read-only only when it can parse the complete command with its supported shell grammar and verify every command and argument against the read-only policy. Any unsupported or ambiguous syntax requires approval. In Plan Mode, the same result is denied instead of executed.

## Supported shell grammar

The analyzer accepts plain commands joined by `|`, `&&`, `||`, `;`, or newlines. On POSIX shells, single quotes, double quotes, and ordinary backslash escapes are recognized so control operators inside an argument are not mistaken for command boundaries. On Windows, the parser follows the narrower `cmd.exe` boundary: single quotes and `%`, `!`, or `^` expansion syntax require approval, while backslashes remain path characters.

The following constructs are outside the automatically approved subset:

- input and output redirection, including heredocs and here-strings;
- command, parameter, process, pathname, and backtick expansion;
- background execution, grouping, comments, and line continuation;
- malformed quoting or control operators;
- shell interpreters, inline scripts, and commands outside the allowlist.

## Command policy

The analyzer allows `ls`, `cat`, `grep`, `pwd`, `which`, `head`, `tail`, and `wc` after the shell grammar check succeeds. Commands with arguments that can write files or execute helpers receive additional validation:

| Command | Automatically approved scope |
| --- | --- |
| `git` | `status`, `log`, `diff`, and `show`, excluding output and external helper options |
| `find` | Queries that do not use deleting, executing, confirmation, or file-output actions |
| `rg` | Searches that do not configure preprocessors or hostname executables |
| `fd` | Searches that do not use per-result or batch execution |
| `sed` | Display operations and substitutions without in-place output, file-loaded programs, write flags, or execution flags |

Long-option abbreviations for unsafe options are treated as unsafe. If a new option cannot be classified confidently, it remains outside the automatic approval path.

## Permission behavior

Explicit Bash deny rules are evaluated before read-only approval in every permission mode. A command classified as read-only is automatically allowed in Default, Plan, and Auto modes. A command that is not classified as read-only follows the normal confirmation or Auto Mode classifier path; Plan Mode denies it.

| Command | Classification |
| --- | --- |
| `git status --short` | Read-only |
| `cat README.md | grep Easy` | Read-only |
| `rg 'foo|bar' src` | Read-only |
| `cat source > target` | Requires approval; denied in Plan Mode |
| `find . -delete` | Requires approval; denied in Plan Mode |
| `sed --in-place 's/a/b/' file` | Requires approval; denied in Plan Mode |
| `cat $(touch target)` | Requires approval; denied in Plan Mode |

Run `npm run test:bash-readonly` after changing the parser, command allowlist, validators, or permission routing. Add a regression case before extending the accepted grammar or command policy.
