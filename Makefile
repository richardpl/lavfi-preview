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
PKG_CONFIG_FLAGS = --shared
EXE = lavfi-preview
IMGUI_DIR = imgui
IMNODES_DIR = imnodes
GLAD_DIR = glad
SOURCES = main.cpp
SOURCES += $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp
SOURCES += $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp
SOURCES += $(IMNODES_DIR)/imnodes.cpp
SOURCES += $(GLAD_DIR)/src/glad.c
OBJS = $(addsuffix .o, $(basename $(notdir $(SOURCES))))
UNAME_S := $(shell uname -s)

CXXFLAGS ?= -g -Wall -Wformat -std=c++17
CXXFLAGS += -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends -I$(IMNODES_DIR) -I$(GLAD_DIR)/include -I/opt/homebrew/include -I/opt/homebrew/Cellar/glfw/3.4/include -I/opt/homebrew/opt/openal-soft/include
CFLAGS ?= -g -Wall -Wformat
CFLAGS += -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends -I$(IMNODES_DIR) -I$(GLAD_DIR)/include -I/opt/homebrew/include -I/opt/homebrew/Cellar/glfw/3.4/include -I/opt/homebrew/opt/openal-soft/include
LIBS ?= -L/usr/local/lib -L/opt/homebrew/lib -L/opt/homebrew/Cellar/glfw/3.4/lib -lglfw -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo -framework OpenAL -lavutil -lavcodec -lavformat -lswresample -lswscale -lavfilter -lavdevice

##---------------------------------------------------------------------
## BUILD FLAGS PER PLATFORM
##---------------------------------------------------------------------

ifeq ($(UNAME_S), Linux) #LINUX
	LIBS += `pkg-config --with-path=$(PKG_CONFIG_PATH) $(PKG_CONFIG_FLAGS) --libs glfw3 libavutil libavcodec libavformat libswresample libswscale libavfilter libavdevice openal`
	CXXFLAGS += `pkg-config --with-path=$(PKG_CONFIG_PATH) $(PKG_CONFIG_FLAGS) --cflags glfw3 libavutil libavcodec libavformat libswresample libswscale libavfilter libavdevice openal`
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

%.o:$(GLAD_DIR)/src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

all: $(EXE)
	@echo Build complete for lavfi-preview

$(EXE): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

clean:
	rm -f $(EXE) $(OBJS)
