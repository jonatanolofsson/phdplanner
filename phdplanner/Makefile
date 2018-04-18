profile:
	tup phdplanner.cpython-36m-x86_64-linux-gnu.so
	valgrind --callgrind-out-file=callgrind.out --tool=callgrind python test_planner.py
	callgrind_annotate --auto=yes callgrind.out

video:
	ffmpeg -r 3 -i planner_%05d.png -vf scale=1280:-2  -c:v libx264 -vf fps=25 -vf scale=1280:-2 -pix_fmt yuv420p out.mp4
