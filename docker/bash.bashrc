export PS1="\[\e[31m\]docker\[\e[m\] \[\e[33m\]\w\[\e[m\] > "
export TERM=xterm-256color
alias grep="grep --color=auto"
alias ls="ls --color=auto"

echo -e "\e[1;31m"
cat<<DBG
=========================================================================
             ___             _____          __       _             
            / _ \___ _  __  / ___/__  ___  / /____ _(_)__  ___ ____
           / // / -_) |/ / / /__/ _ \/ _ \/ __/ _ `/ / _ \/ -_) __/
          /____/\__/|___/  \___/\___/_//_/\__/\_,_/_/_//_/\__/_/   
                 /  _/__  (_) /_(_)__ _/ /____ ___/ /              
                _/ // _ \/ / __/ / _ `/ __/ -_) _  /               
               /___/_//_/_/\__/_/\_,_/\__/\__/\_,_/                
                                                                  
                      GOOD LUCK ON YOUR MISSION
                      
maintainer: Samyakh Tukra
=========================================================================
          
DBG
echo -e "\e[0;33m"

if [[ $EUID -eq 0 ]]; then
  cat <<WARN
WARNING: You are running this container as root, which can cause new files in
mounted volumes to be created as the root user on your host machine.

To avoid this, run the container by specifying your user's userid:

$ docker run -u \$(id -u):\$(id -g) args...
WARN
else
  cat <<EXPL
You are running this container as user with ID $(id -u) and group $(id -g),
which should map to the ID and group for your user on the Docker host. Great!
EXPL
fi

# Turn off colors
echo -e "\e[m"