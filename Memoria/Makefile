# -----------------------
# Plantilla para TFG / TFM 
#
# by Francisco de Sande <fsande@ull.es>
# date: October 24, 2023
# ---

LATEX    = latex
BIBTEX   = bibtex
PDFLATEX = pdflatex
L2HTML   = latex2html
DVIPS    = dvips
BASURA   = basura* *.dvi *.bbl *.aux *.toc *.lof *.lot *.lol *.log *.blg *.out *~ *.pdf *.ps *.backup *.pdf.old
MASTER   = memoria-TFG-AFontenla.tex
SRC      =                                              \
						capitulos/Cap0_Preface.tex 			            \
						capitulos/Cap1_Goals.tex                    \
						capitulos/Cap2_Related_Technologies.tex     \
						capitulos/Cap6_Conclusions_And_Future_Lines_Of_Work.tex 
					
DVI = $(MASTER:.tex=.dvi)
BIB = $(MASTER:.tex=.bib)
PS  = $(MASTER:.tex=.ps)
PDF = $(MASTER:.tex=.pdf)
MASTERSIN =  $(MASTER:.tex=)

all: 
	touch memoria-TFG-AFontenla.`date +%Y-%m-%d`.pdf
	$(MAKE) pdf
	mv memoria-TFG-AFontenla.`date +%Y-%m-%d`.pdf memoria-TFG-AFontenla.`date +%Y-%m-%d`.pdf.old
	cp memoria-TFG-AFontenla.pdf memoria-TFG-AFontenla.`date +%Y-%m-%d`.pdf

HTML: $(PDF)
	$(L2HTML) -dir HTML -split 4 -local_icons -long_titles 20 $(MASTER)

pdf: $(MASTER) $(SRC)
	$(PDFLATEX) $(MASTER) && $(MAKE) bib && $(PDFLATEX) $(MASTER) && $(PDFLATEX) $(MASTER)

ps: $(MASTER) $(SRC) $(DVI)
	$(DVIPS) -o $(PS) $(DVI)

bib: bibliografia.bib
	$(BIBTEX) $(MASTERSIN)

$(DVI): $(MASTER) $(SRC)
	$(LATEX) $(MASTER); $(MAKE) bib; $(LATEX) $(MASTER); $(LATEX) $(MASTER)   

clean:
	$(RM) capitulos/*.aux basura* *.dvi *.bbl *.aux *.toc *.lof *.lot *.log *.blg *.lol *.brf *~ *.out HTML/*

delete:
	$(RM) -R $(BASURA);

mrproper:
	$(MAKE) delete
	cd capitulos;    $(RM) -R $(BASURA);  \

#.SUFFIXES: .tex .dvi. .ps .pdf
#
#%.dvi: %.tex
#	$(LATEX) $(SRC)
#
# end
# --
