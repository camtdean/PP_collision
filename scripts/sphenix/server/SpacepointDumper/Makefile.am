AUTOMAKE_OPTIONS = foreign

AM_CPPFLAGS = \
  -I$(includedir) \
  -isystem$(OFFLINE_MAIN)/include \
  -isystem$(ROOTSYS)/include

AM_LDFLAGS = \
  -L$(libdir) \
  -L$(OFFLINE_MAIN)/lib \
  -L$(OFFLINE_MAIN)/lib64

pkginclude_HEADERS = \
  SpacepointDumper.h

lib_LTLIBRARIES = \
  libspacepointdumper.la

libspacepointdumper_la_SOURCES = \
  SpacepointDumper.cc

libspacepointdumper_la_LIBADD = \
  -lphool \
  -lSubsysReco \
  -lg4eval \
  -lphg4hit \
  -ltrack_io \
  -lglobalvertex_io

BUILT_SOURCES = testexternals.cc

noinst_PROGRAMS = \
  testexternals

testexternals_SOURCES = testexternals.cc
testexternals_LDADD = libspacepointdumper.la

testexternals.cc:
	echo "//*** this is a generated file. Do not commit, do not edit" > $@
	echo "int main()" >> $@
	echo "{" >> $@
	echo "  return 0;" >> $@
	echo "}" >> $@

clean-local:
	rm -f $(BUILT_SOURCES)
