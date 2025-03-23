## ML END TO END CODE

This is about how i create an ml project an how the modules and project structure looks like.

Initially i create a github repository as a generic one(which don't have anything in it)
then craete file in my localmechine c: user-> mlproject-> anacoda prompt-> cd to c:user/mlproject/ -> code . --> opened vs in that location directory --> vs terminal -> vscs (conda create -p venv(foldername) python== 3.8 (version) -y (verfication yes))--> conda activate venu/ (activate ever enviroment) -->  
now sync the git hub with project using github quick setup --> git init
git add README.md--> git commit -m "first commit" --> git branch M main
git remote and orgin user git repo link --> git push -u orgin main
create .gitignore (used to some of file not required files will removed.) in git hub reposity
then pull all the request using git pull

## we will automatic this using some commands

## setup.py is used for packaging and distributing python prjoects.(with this we can able to build my ml prject as a package and even i can deploy in python pipi them in other envioments)

( find_packages--> find requirement packages related to this prjoect)

## requiments.txt list dependences needed to run the prjoects.

after setup all thing we do something called pip install -r requirements.txt --> created mltest.egg-info file which we can deploy in pipi
