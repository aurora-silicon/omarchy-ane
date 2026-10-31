PYTHON ?= python3
CC ?= cc
PREFIX ?= /usr
LIBDIR ?= $(PREFIX)/lib
PYTHON_DIR ?= $(LIBDIR)/aurora-camera-effects
CFLAGS += -std=c11 -O2 -Wall -Wextra -Werror -fPIC
CPPFLAGS += $(shell $(PYTHON) -c 'import sysconfig; print("-I"+sysconfig.get_path("include"))')
PYLIBS = $(shell $(PYTHON) -c 'import sysconfig; print("-L"+sysconfig.get_config_var("LIBDIR"), "-lpython"+sysconfig.get_python_version(), "-Wl,-rpath,"+sysconfig.get_config_var("LIBDIR"))')
PYSHARED = $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR")+"/"+(sysconfig.get_config_var("INSTSONAME") or sysconfig.get_config_var("LDLIBRARY")))')
PYEXEC = $(shell $(PYTHON) -c 'import sys; print(sys.executable)')
.PHONY: all install clean
all: libaurora-camera-effects.so
libaurora-camera-effects.so: camera_effects.c camera_effects.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -DCE_PYTHON_DIR='"$(PYTHON_DIR)"' -DCE_PYTHON_EXECUTABLE='"$(PYEXEC)"' -DCE_PYTHON_LIBRARY='"$(PYSHARED)"' -shared $< -o $@ $(PYLIBS) -pthread -ldl
install: all
	install -d $(DESTDIR)$(LIBDIR)/aurora-camera-denoise $(DESTDIR)$(PYTHON_DIR) $(DESTDIR)$(PREFIX)/include
	install libaurora-camera-effects.so $(DESTDIR)$(LIBDIR)/aurora-camera-denoise/
	install -m644 camera_effects.h $(DESTDIR)$(PREFIX)/include/
	cp -r aurora_camera_effects models $(DESTDIR)$(PYTHON_DIR)/
	install -m644 README.md NOTICES.md LICENCE-AUDIT.md $(DESTDIR)$(PYTHON_DIR)/
	install -m644 ../../LICENSE $(DESTDIR)$(PYTHON_DIR)/LICENSE
clean:
	rm -f libaurora-camera-effects.so
