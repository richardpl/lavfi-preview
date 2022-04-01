#
# Cross Platform Makefile
# Compatible with MSYS2/MINGW, Ubuntu 14.04.1 and Mac OS X
#
# You will need GLFW (http://www.glfw.org):
# Linux:
#   apt-get install libglfw-dev
# Mac OS X:
#   brew install glfw
# MSYS2:
#   pacman -S --noconfirm --needed mingw-w64-x86_64-toolchain mingw-w64-x86_64-glfw
#

#CXX = g++
#CXX = clang++

PKG_CONFIG_PATH = /usr/local/lib/pkgconfig
EXE = lavfi-preview
IMGUI_DIR = imgui
IMNODES_DIR = imnodes
SOURCES = main.cpp
SOURCES += $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp
SOURCES += $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp
SOURCES += $(IMNODES_DIR)/imnodes.cpp
OBJS = $(addsuffix .o, $(basename $(notdir $(SOURCES))))
UNAME_S := $(shell uname -s)
LINUX_GL_LIBS = -lGL

CXXFLAGS = -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends -I$(IMNODES_DIR)
CXXFLAGS += -g -Wall -Wformat
LIBS =

##---------------------------------------------------------------------
## BUILD FLAGS PER PLATFORM
##---------------------------------------------------------------------

ifeq ($(UNAME_S), Linux) #LINUX
	LIBS += $(LINUX_GL_LIBS) `pkg-config --with-path=$(PKG_CONFIG_PATH) --shared --libs glfw3`
	LIBS += `pkg-config --with-path=$(PKG_CONFIG_PATH) --shared --libs libavutil`
	LIBS += `pkg-config --with-path=$(PKG_CONFIG_PATH) --shared --libs libavcodec`
	LIBS += `pkg-config --with-path=$(PKG_CONFIG_PATH) --shared --libs libavformat`
	LIBS += `pkg-config --with-path=$(PKG_CONFIG_PATH) --shared --libs libswresample`
	LIBS += `pkg-config --with-path=$(PKG_CONFIG_PATH) --shared --libs libswscale`
	LIBS += `pkg-config --with-path=$(PKG_CONFIG_PATH) --shared --libs libavfilter`
	LIBS += `pkg-config --with-path=$(PKG_CONFIG_PATH) --shared --libs openal`

	CXXFLAGS += `pkg-config --with-path=$(PKG_CONFIG_PATH) --cflags glfw3`
	CXXFLAGS += `pkg-config --with-path=$(PKG_CONFIG_PATH) --cflags libavutil`
	CXXFLAGS += `pkg-config --with-path=$(PKG_CONFIG_PATH) --cflags libavcodec`
	CXXFLAGS += `pkg-config --with-path=$(PKG_CONFIG_PATH) --cflags libavformat`
	CXXFLAGS += `pkg-config --with-path=$(PKG_CONFIG_PATH) --cflags libswresample`
	CXXFLAGS += `pkg-config --with-path=$(PKG_CONFIG_PATH) --cflags libswscale`
	CXXFLAGS += `pkg-config --with-path=$(PKG_CONFIG_PATH) --cflags libavfilter`
	CXXFLAGS += `pkg-config --with-path=$(PKG_CONFIG_PATH) --cflags openal`
	CFLAGS = $(CXXFLAGS)
endif

##---------------------------------------------------------------------
## BUILD RULES
##---------------------------------------------------------------------

%.o:%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o:$(IMGUI_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o:$(IMGUI_DIR)/backends/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o:$(IMNODES_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

all: $(EXE)
	@echo Build complete for lavfi-preview

$(EXE): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

clean:
	rm -f $(EXE) $(OBJS)
