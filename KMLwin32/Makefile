#-----------------------------------------------------------------------------
# KMlocal: A testbed for k-means clustering algorithms based on local search
# Version: 1.7
#-----------------------------------------------------------------------------
# Copyright (c) 2004-2005 David M. Mount and the University of Maryland
# All Rights Reserved.
#
# PLEASE READ THE FILE "Copyright.txt" FOR COPYRIGHT INFORMATION AND
# DISCLAIMER.
#-----------------------------------------------------------------------------
# Makefile for Windows Versions.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# This is not used for compiling the dll. It is just used for cleaning
# things up for distribution.  For compilcation, open the Ann.sln
# solution file in Microsoft Windows Visual Studio.NET.
#-----------------------------------------------------------------------------

default:
	@echo "Enter one of the following:"
	@echo "  make clean                remove object files"
	@echo "  make realclean            remove library and executable files"
	@echo " "
	@echo "See file Makefile for other compilation options."

BASE = KMLwin32

#-----------------------------------------------------------------------------
# Remove .o files and core files
#-----------------------------------------------------------------------------
clean:
	-rm -f -r kmlminimal/Debug kmlminimal/Release kmlsample/Debug kmlsample/Release kmltest/Debug kmltest/Release

#-----------------------------------------------------------------------------
# Remove everthing that can be remade
#-----------------------------------------------------------------------------
realclean:  clean
	-rm -f $(BASE).ncb $(BASE).suo $(BASE).rc
