#!/bin/bash
ext=$(python3-config --extension-suffix)
inc=$(python3 -m pybind11 --includes)
inc2=$(pkg-config --cflags python3)
echo ':foreach *.cpp | ../<pre> |> $(COMPILER) $(CFLAGS) $(APP_CFLAGS) $(PY_CFLAGS)' "${inc}" "${inc2}" '%f -o %o $(LDFLAGS) $(APP_LDFLAGS) $(PY_LDFLAGS) |>' "%B${ext}" '$(TOP)/<libs>'
