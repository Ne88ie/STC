git filter-branch --commit-filter 'if [ "$GIT_AUTHOR_NAME" = "Moiseeva Anastasia" ];
  then  export GIT_AUTHOR_EMAIL=nast.mois@gmail.com;
        fi; git commit-tree "$@"'
