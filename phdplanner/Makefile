profile:
	tup phdplanner.cpython-36m-x86_64-linux-gnu.so
	valgrind --callgrind-out-file=callgrind.out --tool=callgrind python test_planner.py
	callgrind_annotate --auto=yes callgrind.out
