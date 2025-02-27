#!/bin/bash
# Backup script for AVA6 project

echo "AVA6 Backup Script"
echo "=================="
echo

# Get the current date and time for the commit message
DATE=$(date +"%Y-%m-%d %H:%M:%S")
COMMIT_MSG="Backup: $DATE"

# Check if a custom commit message was provided
if [ $# -gt 0 ]; then
    COMMIT_MSG="$1"
fi

echo "Using commit message: '$COMMIT_MSG'"
echo

# Add all changes
echo "Adding all changes..."
git add .

# Commit changes
echo "Committing changes..."
git commit -m "$COMMIT_MSG"

# Push to GitHub
echo "Pushing to GitHub..."
git push origin develop

echo
echo "Backup completed successfully!"
echo 