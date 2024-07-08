@echo off

:: the plain name of the *.tex file that you want to compile (without extension)
set document=BEARBEITER

:: clean up deprecated compilation files
call:cleanup
:: close acrobat reader
tskill acrobat  
:: first time run pdflatex to generate aux file and resolve dependencies
pdflatex %document%.tex
:: bibtex document
bibtex %document%
:: pdflatex again twice to include the updated bibliography
pdflatex %document%.tex
pdflatex %document%.tex
:: if package in use: make index
:: makeindex.exe %document%.nlo -s nomencl.ist -o %document%.nls
:: pdflatex %document%.tex
:: open result with acrobat reader
START "" %document%.pdf
:: clean up files again
call:cleanup

:cleanup
del *.log
del *.dvi
del *.aux
del *.bbl
del *.blg
del *.brf
del *.out
del *.lof
del *.toc
del *.idx
goto:eof