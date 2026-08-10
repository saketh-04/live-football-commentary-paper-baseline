.PHONY: prepare build install

prepare:
	sshfs peter:/raid_elmo/home/lr/moriy/SoccerNet ./SoccerNet_in_lrlab -o volname=userdocs,reconnect,IdentityFile=/Users/heste/.ssh/peter

install:
	npm install

build: install prepare
	npm run build

serve: build
	npm run start
