---
name: project-continuity-skill
description: Setup project continuity with git+backup workflow for any development project
version: 1.0.0
author: Glenn Highcove and OpenCode Agent via Vibe Coding
license: MIT
readme: README.md
homepage: https://github.com/ghighcove/yacht-market-analysis
repository: https://github.com/ghighcove/yacht-market-analysis.git
categories: [development, productivity, project-management]
tags: [git, backup, workflow, automation]
compatibility: opencode
metadata: {
  "audience": "maintainers", 
  "workflow": "github"
}
---

# Project Continuity Framework

Creates consistent git+backup workflow with session tracking for any development project.

## What I do

- Automated project structure creation
- Git repository initialization (private by default, public when ready)
- Session status checking and documentation
- Automatic backup system (daily + on-demand)
- Progress tracking and recovery protocols
- Template-based README generation
- Multi-agent collaboration support
- Unicode-safe cross-platform compatibility

## Usage

```bash
skill({name: "project-continuity-skill"})
```

## File Structure Created

```
project-name/
├── .git/                   # Git configuration
├── data/                   # Datasets and sources  
├── scripts/                 # Automation utilities
├── docs/                    # Documentation and notes
├── backup/                  # Versioned backups
├── config/                   # Configuration files
└── project-tracker.md       # Session continuity log
```

## Scripts Generated

- `scripts/session_status.py` - Check current session state
- `scripts/end_session.py` - Document session completion  
- `scripts/backup_project.py` - Create timestamped backups
- `scripts/setup_structure.py` - Initialize directories

## Documentation

- `README.md` - Project overview and getting started guide
- `docs/workflow-guide.md` - Development workflow instructions
- `project-tracker.md` - Session progress and recovery log

## Git Workflow

1. **Development Phase**: Private repository, frequent commits
2. **Backup Strategy**: Local + remote backup when ready
3. **Release Phase**: Public repository when stable
4. **Recovery Protocol**: Step-by-step instructions if repository lost

## Session Management

- Each session documented with start/end times
- Work tracked with file changes and decisions
- Automatic progress notes and backup creation
- Seamless handoff between different developers/agents

## Compatibility

- Cross-platform (Windows, macOS, Linux)
- Python 3.8+ compatible
- Unicode-safe output and file operations
- Error handling for common development issues

## Benefits

- **Zero Setup Time**: Projects ready in 60 seconds
- **Perfect Continuity**: No work ever lost between sessions
- **Professional Standards**: Consistent structure and documentation
- **Team Collaboration**: Multiple developers can work seamlessly
- **Disaster Recovery**: Complete restoration protocols