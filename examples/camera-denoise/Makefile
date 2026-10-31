PYTHON ?= python3
CC ?= cc
PREFIX ?= /usr
PYTHON_DIR ?= $(PREFIX)/lib/aurora-camera-ane
CPPFLAGS += $(shell $(PYTHON) -c 'import sysconfig; print("-I"+sysconfig.get_path("include"))')
CFLAGS += -std=c11 -O2 -Wall -Wextra -Werror -fPIC
PYLIBS = $(shell $(PYTHON) -c 'import sysconfig; print("-L"+sysconfig.get_config_var("LIBDIR"), "-lpython"+sysconfig.get_python_version(), "-Wl,-rpath,"+sysconfig.get_config_var("LIBDIR"))')
PYEXEC = $(shell $(PYTHON) -c 'import sys; print(sys.executable)')

.PHONY: all check clean install
all: libcamera-ane-backend.so
libcamera-ane-backend.so: camera_ane_backend.c camera_ane_backend.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -DCAMERA_ANE_PYTHON_DIR='"$(PYTHON_DIR)"' -DCAMERA_ANE_PYTHON_EXECUTABLE='"$(PYEXEC)"' -shared $< -o $@ $(PYLIBS) -pthread
test-bridge: tests/test_bridge.c libcamera-ane-backend.so
	$(CC) $(CFLAGS) -I. $< -L. -lcamera-ane-backend -Wl,-rpath,'$$ORIGIN' -o $@
check: test-bridge
	./test-bridge
install: libcamera-ane-backend.so
	install -d $(DESTDIR)$(PYTHON_DIR) $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include
	install libcamera-ane-backend.so $(DESTDIR)$(PREFIX)/lib/
	install camera_ane_backend.h $(DESTDIR)$(PREFIX)/include/
	cp -r aurora_camera_ane models $(DESTDIR)$(PYTHON_DIR)/
clean:
	rm -f libcamera-ane-backend.so test-bridge
