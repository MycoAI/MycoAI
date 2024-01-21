import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import requests

from loggingwrapper import LoggingWrapper

if sys.version_info[0] >= 3:
    unicode = str

import os
from Bio import SeqIO
import json

class RDPClassifier():

    def downloadClassifier(self):
        rdpclassifierpath = str(Path(__file__).parent.absolute())
        response = requests.get('https://sourceforge.net/projects/rdp-classifier/files/latest/download')

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            zip_file_path = os.path.join(rdpclassifierpath, 'rdpclassifier.zip')

            with open(zip_file_path, 'wb') as zip_file:
                zip_file.write(response.content)

            # Step 2: Extract the ZIP file
            with ZipFile(zip_file_path, 'r') as zip_ref:
                unzipped_dir = os.path.join(rdpclassifierpath, 'rdpclassifier_unzipped')
                zip_ref.extractall(unzipped_dir)

                # Step 3: Locate and move the classifier.jar file
                classifier_path = os.path.join(unzipped_dir, 'dist', 'classifier.jar')

                # Check if the file exists before moving
                if os.path.exists(classifier_path):
                    # Optionally, you can move the classifier.jar to a different directory
                    destination_dir = rdpclassifierpath
                    destination_path = os.path.join(destination_dir, 'classifier.jar')
                    os.rename(classifier_path, destination_path)

                    # Step 4: Remove the ZIP file and unzipped directory
                    os.remove(zip_file_path)
                    os.rmdir(unzipped_dir)
                else:
                    LoggingWrapper.error("The classifier.jar file was not found in the downloaded ZIP file.", color="red", bold=True)
                    sys.exit(1)

        else:
            LoggingWrapper.error("Unable to download the file. Status code: " + str(response.status_code), color="red", bold=True)
    def GetBase(self, filename):
        return filename[:-(len(filename) - filename.rindex("."))]

    def GetUnidentifiedSpeciesName(self, seqid):
        name = "unidentified_species_of_" + seqid
        return name

    def GetUnidentifiedGenusName(self, seqid, species):
        name = ""
        if species != "":
            name = "unidentified_genus_of_" + species
        else:
            name = "unidentified_genus_of_" + seqid
        return name

    def GetUnidentifiedFamilyName(self, seqid, species, genus):
        name = ""
        if genus != "":
            name = "unidentified_family_of_" + genus
        elif species != "":
            name = "unidentified_family_of_" + species
        else:
            name = "unidentified_family_of_" + seqid
        return name

    def GetUnidentifiedOrderName(self, seqid, species, genus, family):
        name = ""
        if family != "":
            name = "unidentified_order_of_" + family
        elif genus != "":
            name = "unidentified_order_of_" + genus
        elif species != "":
            name = "unidentified_order_of_" + species
        else:
            name = "unidentified_order_of_" + seqid
        return name

    def GetUnidentifiedClassName(self, seqid, species, genus, family, order):
        name = ""
        if order != "":
            name = "unidentified_class_of_" + order
        elif family != "":
            name = "unidentified_class_of_" + family
        elif genus != "":
            name = "unidentified_class_of_" + genus
        elif species != "":
            name = "unidentified_class_of_" + species
        else:
            name = "unidentified_class_of_" + seqid
        return name

    def GetUnidentifiedPhylumName(self, seqid, species, genus, family, order, bioclass):
        name = ""
        if bioclass != "":
            name = "unidentified_phylum_of_" + bioclass
        elif order != "":
            name = "unidentified_phylum_of_" + order
        elif family != "":
            name = "unidentified_phylum_of_" + family
        elif genus != "":
            name = "unidentified_phylum_of_" + genus
        elif species != "":
            name = "unidentified_phylum_of_" + species
        else:
            name = "unidentified_phylum_of_" + seqid
        return name

    def GetUnidentifiedKingdomName(self, seqid, species, genus, family, order, bioclass, phylum):
        name = ""
        if phylum != "":
            name = "unidentified_kingdom_of_" + phylum
        elif bioclass != "":
            name = "unidentified_kingdom_of_" + bioclass
        elif order != "":
            name = "unidentified_kingdom_of_" + order
        elif family != "":
            name = "unidentified_kingdom_of_" + family
        elif genus != "":
            name = "unidentified_kingdom_of_" + genus
        elif species != "":
            name = "unidentified_kingdom_of_" + species
        else:
            name = "unidentified_kingdom_of_" + seqid
        return name

    def LoadClassification(self, seqids, classificationfilename, pos):
        kingdompos = 1
        phylumpos = 2  # the position of the phyla in the classification file
        classpos = 3
        orderpos = 4
        familypos = 5
        genuspos = 6
        speciespos = 7
        species = [""] * len(seqids)
        genera = [""] * len(seqids)
        families = [""] * len(seqids)
        orders = [""] * len(seqids)
        classes = [""] * len(seqids)
        phyla = [""] * len(seqids)
        kingdoms = [""] * len(seqids)
        classificationfile = open(classificationfilename)
        classifications = [""] * len(seqids)
        labels = [""] * len(seqids)
        level = 0  # kingdom
        sep = "\t"
        # load species, genera, families, orders, classes, phyla, kingdoms for sequences
        for line in classificationfile:
            line = unicode(line, errors='ignore')
            if ";" in line:
                sep = ";"
            words = line.split(sep)
            if line.startswith("#"):
                i = 0
                for word in words:
                    word = word.lower().rstrip()
                    if word == "species":
                        speciespos = i
                    if word == "genus":
                        genuspos = i
                    if word == "family":
                        familypos = i
                    if word == "order":
                        orderpos = i
                    if word == "class":
                        classpos = i
                    if word == "phylum":
                        phylumpos = i
                    if word == "kingdom":
                        kingdompos = i
                    i = i + 1
                if pos == 0:
                    pos = speciespos
                continue
            kingdom = ""
            phylum = ""
            bioclass = ""
            order = ""
            family = ""
            genus = ""
            spec = ""
            if kingdompos > 0 and len(words) > kingdompos:
                kingdom = words[kingdompos].rstrip()
            if phylumpos > 0 and len(words) > phylumpos:
                phylum = words[phylumpos].rstrip()
            if classpos > 0 and len(words) > classpos:
                bioclass = words[classpos].rstrip()
            if orderpos > 0 and len(words) > orderpos:
                order = words[orderpos].rstrip()
            if familypos > 0 and len(words) > familypos:
                family = words[familypos].rstrip()
            if genuspos > 0 and len(words) > genuspos:
                genus = words[genuspos].rstrip()
            if speciespos > 0 and len(words) > speciespos:
                spec = words[speciespos].rstrip()
            seqid = words[0].replace(">", "").rstrip()
            if kingdom == "":
                kingdom = self.GetUnidentifiedKingdomName(seqid, spec, genus, family, order, bioclass, phylum)
            if phylum == "":
                phylum = self.GetUnidentifiedPhylumName(seqid, spec, genus, family, order, bioclass)
            if bioclass == "":
                bioclass = self.GetUnidentifiedClassName(seqid, spec, genus, family, order)
            if order == "":
                order = self.GetUnidentifiedOrderName(seqid, spec, genus, family)
            if family == "":
                family = self.GetUnidentifiedFamilyName(seqid, spec, genus)
            if genus == "":
                genus = self.GetUnidentifiedGenusName(seqid, spec)
            if spec == "":
                spec = self.GetUnidentifiedSpeciesName(seqid)
            label = ""
            if pos == speciespos:
                level = 6
                label = spec
                rank = "species"
            if pos == genuspos:
                level = 5
                label = genus
                rank = "genus"
            if pos == familypos:
                level = 4
                label = family
                rank = "family"
            if pos == orderpos:
                level = 3
                label = order
                rank = "order"
            if pos == classpos:
                level = 2
                label = bioclass
                rank = "class"
            if pos == phylumpos:
                level = 1
                label = phylum
                rank = "phylum"
            if pos == kingdompos:
                level = 1
                label = kingdom
                rank = "kingdom"
            if seqid in seqids:
                index = seqids.index(seqid)
                classification = "Root"
                kingdom = "Fungi"
                if level >= 0:
                    kingdoms[index] = kingdom
                    classification = classification + ";" + kingdom
                if level >= 1:
                    phyla[index] = phylum
                    classification = classification + ";" + phylum
                if level >= 2:
                    classes[index] = bioclass
                    classification = classification + ";" + bioclass
                if level >= 3:
                    orders[index] = order
                    classification = classification + ";" + order
                if level >= 4:
                    families[index] = family
                    classification = classification + ";" + family
                if level >= 5:
                    genera[index] = genus
                    classification = classification + ";" + genus
                if level >= 6:
                    species[index] = spec
                    classification = classification + ";" + spec
                classifications[index] = classification
                labels[index] = label
        return species, genera, families, orders, classes, phyla, kingdoms, classifications, labels, rank

    def GenerateTaxaIDs(self, species, genera, families, orders, classes, phyla, kingdoms, taxaidfilename):
        taxids = []
        taxa = []
        parenttaxids = []
        depths = []
        ranks = []
        # add root
        taxids.append(0)
        taxa.append("Root")
        parenttaxids.append(-1)
        depths.append(0)
        ranks.append("rootrank")
        i = 0
        for kingdom in kingdoms:
            # add  kingdom
            kingdom_id = len(taxa)
            if kingdom != "":
                if kingdom in taxa:
                    kingdom_id = taxa.index(kingdom)
                else:
                    taxids.append(len(taxa))
                    taxa.append(kingdom)
                    parenttaxids.append(0)
                    depths.append(1)
                    ranks.append("domain")
            phylum_id = len(taxa)
            if len(phyla) > 0:
                if phyla[i] in taxa:
                    phylum_id = taxa.index(phyla[i])
                elif phyla[i] != "":
                    taxids.append(len(taxa))
                    taxa.append(phyla[i])
                    parenttaxids.append(kingdom_id)
                    depths.append(2)
                    ranks.append("phylum")
            class_id = len(taxa)
            if len(classes) > i:
                if classes[i] in taxa:
                    class_id = taxa.index(classes[i])
                elif classes[i] != "":
                    taxids.append(len(taxa))
                    taxa.append(classes[i])
                    parenttaxids.append(phylum_id)
                    depths.append(3)
                    ranks.append("class")
            order_id = len(taxa)
            if len(orders) > i:
                if orders[i] in taxa:
                    order_id = taxa.index(orders[i])
                elif orders[i] != "":
                    taxids.append(len(taxa))
                    taxa.append(orders[i])
                    parenttaxids.append(class_id)
                    depths.append(4)
                    ranks.append("order")
            family_id = len(taxa)
            if len(families) > i:
                if families[i] in taxa:
                    family_id = taxa.index(families[i])
                elif families[i] != "":
                    taxids.append(len(taxa))
                    taxa.append(families[i])
                    parenttaxids.append(order_id)
                    depths.append(5)
                    ranks.append("family")
            genus_id = len(taxa)
            if len(genera) > i:
                if genera[i] in taxa:
                    genus_id = taxa.index(genera[i])
                elif genera[i] != "":
                    taxids.append(len(taxa))
                    taxa.append(genera[i])
                    parenttaxids.append(family_id)
                    depths.append(6)
                    ranks.append("genus")
            species_id = len(taxa)
            if len(species) > i:
                if species[i] in taxa:
                    species_id = taxa.index(species[i])
                elif species[i] != "":
                    taxids.append(len(taxa))
                    taxa.append(species[i])
                    parenttaxids.append(genus_id)
                    depths.append(7)
                    ranks.append("species")
            i = i + 1
        # write to taxaidfilename
        taxaidfile = open(taxaidfilename, "w")
        i = 0
        for taxid in taxids:
            taxaidfile.write(
                str(taxid) + "*" + taxa[i] + "*" + str(parenttaxids[i]) + "*" + str(depths[i]) + "*" + ranks[i] + "\n")
            i = i + 1
        taxaidfile.close()

    def GenerateRDFFastaFile(self, seqids, labels, classifications, trainfastafilename, rdpfastafilename):
        classdict = {}
        rdpfastafile = open(rdpfastafilename, "w")
        trainfastafile = open(trainfastafilename)
        seqid = ""
        seq = ""
        for line in trainfastafile:
            if line.startswith(">"):
                # add the previous sequences to classes
                seqid = line.rstrip().replace(">", "")
                i = seqids.index(seqid)
                label = labels[i]
                classification = classifications[i]
                if label in classdict.keys():
                    if len(classification) > len(classdict[label]['classification']):
                        classdict[label]['classification'] = classification
                    classdict[label]['seqids'].append(seqid)
                else:
                    classdict.setdefault(label, {})
                    classdict[label]['classification'] = classification
                    classdict[label]['seqids'] = [seqid]
                rdpfastafile.write(">" + seqid + "\t" + classification + "\n")
            else:
                seq = seq + line.rstrip()
                rdpfastafile.write(line)
        rdpfastafile.close()
        trainfastafile.close()
        return classdict

    def GetSeqIDs(self, seqrecords):
        seqids = []
        for seqrecord in seqrecords:
            seqids.append(seqrecord.id)
        return seqids

    def SaveClasses(self, jsonfilename, classdict):
        # write json dict
        with open(jsonfilename, "w") as json_file:
            json.dump(classdict, json_file, encoding='latin1')

    def SaveConfig(self, configfilename, classifiername, fastafilename, jsonfilename, classificationfilename,
                   classificationpos, rdpfastafilename, rdpidfilename):
        if not classifiername.startswith("/"):
            classifiername = os.getcwd() + "/" + classifiername
        if not fastafilename.startswith("/"):
            fastafilename = os.getcwd() + "/" + fastafilename
        if not classificationfilename.startswith("/"):
            classificationfilename = os.getcwd() + "/" + classificationfilename
        model = "rdp"
        # save the config:classifierfilename, model, classificationfilename,classificationpos,k-mer
        configfile = open(configfilename, "w")
        configfile.write("Model: " + model + "\n")
        configfile.write("Fasta filename: " + fastafilename + "\n")
        configfile.write("Classification filename: " + classificationfilename + "\n")
        configfile.write("Column number to be classified: " + str(classificationpos) + "\n")
        configfile.write("Classes filename: " + jsonfilename + "\n")
        configfile.write("RDP fasta filename: " + rdpfastafilename + "\n")
        configfile.write("RDP ID filename: " + rdpidfilename + "\n")
        configfile.close()




    def LoadConfig(self, modelname):
        if modelname[len(modelname) - 1] == "/":
            modelname = modelname[:-1]
        basename = modelname
        if "/" in modelname:
            basename = modelname[modelname.rindex("/") + 1:]
        configfilename = modelname + "/" + basename + ".config"
        classifiername = ""
        jsonfilename = ""
        classificationfilename = ""
        classificationpos = 0
        configfile = open(configfilename)
        for line in configfile:
            texts = line.split(": ")
            if texts[0] == "Classifier name":
                classifiername = texts[1].rstrip()
            if texts[0] == "Classification filename":
                classificationfilename = texts[1].rstrip()
            if texts[0] == "Column number to be classified":
                classificationpos = int(texts[1].rstrip())
            if texts[0] == "RDP fasta filename":
                rdpfastafilename = texts[1].rstrip()
            if texts[0] == "RDP ID filename":
                rdpidfilename = texts[1].rstrip()
            if texts[0] == "Classes filename":
                jsonfilename = texts[1].rstrip()
        return jsonfilename, classifiername, classificationfilename, classificationpos, rdpfastafilename, rdpidfilename
    def GetPredictedLabels(self, testseqids, rdpclassifiedfilename):
        predlabels = [""] * len(testseqids)
        scores = [0] * len(testseqids)
        ranks = [""] * len(testseqids)
        rdpfile = open(rdpclassifiedfilename)
        for line in rdpfile:
            words = line.split("\t")
            n = len(words)
            seqid = words[0]
            i = testseqids.index(seqid)
            scores[i] = float(words[n - 1].rstrip())
            ranks[i] = words[n - 2]
            predlabels[i] = words[n - 3]
        rdpfile.close()
        return predlabels, scores, ranks

    def SavePrediction(self, classdict, testseqids, testlabels, predlabels, probas, outputname):
        output = open(outputname, "w")
        output.write("SequenceID\tGiven classification\tPrediction\tFull classification of prediction\tProbability\n")
        i = 0
        keys = classdict.keys()
        for i in range(0, len(testseqids)):
            seqid = testseqids[i]
            giventaxonname = testlabels[i]
            predictedname = predlabels[i]
            proba = probas[i]
            # predictedname=predictedname.encode('ascii', 'ignore')
            classification = classdict[predictedname]['classification']
            output.write(
                seqid + "\t" + giventaxonname + "\t" + predictedname + "\t" + classification + "\t" + str(proba) + "\n")
            i = i + 1
        output.close()
    def train(self, args):
        fastafilename= args.input
        classificationfilename=args.classification
        classificationpos=args.classificationpos
        rdpclassifierpath = str(Path(__file__).parent.absolute())
        modelname=args.out
        if (Path(rdpclassifierpath) / "classifier.jar").exists() == False:
            LoggingWrapper.error("The classifier.jar is not found in the folder " + rdpclassifierpath + ".", color="red", bold=True)
            sys.exit(1)

        basefilename = self.GetBase(fastafilename)
        # load seq records
        seqrecords = list(SeqIO.parse(fastafilename, "fasta"))
        # train the data using rdp model
        trainseqids = self.GetSeqIDs(seqrecords)

        # load taxonomic classifcation of the train dataset
        species, genera, families, orders, classes, phyla, kingdoms, classifications, trainlabels, level = self.LoadClassification(
            trainseqids, classificationfilename, classificationpos)

        # save model
        if modelname == None or modelname == "":
            modelname = basefilename + "_rdp_classifier"
            if level != "":
                modelname = basefilename + "_" + level + "_rdp_classifier"
        if os.path.isdir(modelname) == False:
            os.system("mkdir " + modelname)
        basename = modelname
        if "/" in modelname:
            basename = modelname[modelname.rindex("/") + 1:]
        # basename=basefilename + "_" + level + "_rdp_classifier"
        if "/" in basename:
            basename = basename[len(basename) - basename.rindex("/"):]
        # cp_result = subprocess.run("cp " + rdpclassifierpath + "/classifier.jar " + modelname + "/", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', check=True)
        # if cp_result.returncode != 0:
        #     LoggingWrapper.error("Error while copying classifier.jar.", color="red", bold=True)
        #     for line in cp_result.stderr.splitlines():
        #         LoggingWrapper.error(line)
        #     sys.exit(cp_result.returncode)

        # train the rdp classifier
        rdpfastafilename = modelname + "/" + basefilename + ".rdp.fasta"
        rdptaxaidfilename = modelname + "/" + basefilename + ".rdp.tid"

        # generate the taxaid file for the rdp model
        self.GenerateTaxaIDs(species, genera, families, orders, classes, phyla, kingdoms, rdptaxaidfilename)

        # generate fasta file for the rdp model
        classdict = self.GenerateRDFFastaFile(trainseqids, trainlabels, classifications, fastafilename, rdpfastafilename)

        traincommand = "java -Xmx1g -jar " + rdpclassifierpath + "/classifier.jar train -o " + modelname + " -s " + rdpfastafilename + " -t " + rdptaxaidfilename
        traincommand_res = subprocess.run(traincommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', check=True)
        if traincommand_res.returncode != 0:
            LoggingWrapper.error("Error while training the RDP classifier.", color="red", bold=True)
            for line in traincommand_res.stderr.splitlines():
                LoggingWrapper.error(line)
            sys.exit(traincommand_res.returncode)
        LoggingWrapper.info("The RDP classifier is trained.", color="green", bold=True)
        for line in traincommand_res.stdout.splitlines():
            LoggingWrapper.info(line)
        # save seqids for each classification
        jsonfilename = modelname + "/" + basename + ".classes"
        self.SaveClasses(jsonfilename, classdict)
        # save config
        configfilename = modelname + "/" + basename + ".config"
        self.SaveConfig(configfilename, modelname, fastafilename, jsonfilename, classificationfilename, classificationpos,
                   rdpfastafilename, rdptaxaidfilename)

        LoggingWrapper.info("The classifier is saved in the folder " + modelname + ".", color="green", bold=True)
    def classify(self, args):
        testfastafilename = args.input
        rdpclassifierpath = str(Path(__file__).parent.absolute())
        modelname = args.classifier
        rdp_output = args.out
        if (Path(rdpclassifierpath) / "classifier.jar").exists() == False:
            LoggingWrapper.error("The classifier.jar is not found in the folder " + rdpclassifierpath + ".", color="red", bold=True)
            sys.exit(1)
        basefilename = self.GetBase(testfastafilename)

        # load config of the model
        classesfilename, classifiername, classificationfilename, classificationpos, rdpfastafilename, rdpidfilename = self.LoadConfig(
            modelname)

        # load seq records
        seqrecords = list(SeqIO.parse(testfastafilename, "fasta"))
        testseqids = self.GetSeqIDs(seqrecords)

        # load given taxonomic classifcation of the test dataset
        testspecies, testgenera, testfamilies, testorders, testclasses, testphyla, testkingdoms, testclassifications, testlabels, rank = self.LoadClassification(
            testseqids, classificationfilename, classificationpos)

        # run the model
        basename = modelname
        if "/" in basename:
            basename = basename[basename.rindex("/") + 1:]

        rdpclassifiedfilename = basefilename + "." + basename + ".out"
        if rdp_output == None or rdp_output == "":
            rdp_output = basefilename + "." + basename + ".classified"

        testcommand = "java -Xmx1g -jar " + rdpclassifierpath + "/classifier.jar classify -t " + modelname + "/rRNAClassifier.properties -o " + rdpclassifiedfilename + " " + testfastafilename
        testcommand_res = subprocess.run(testcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', check=True)
        if testcommand_res.returncode != 0:
            LoggingWrapper.error("Error while classifying sequences.", color="red", bold=True)
            for line in testcommand_res.stderr.splitlines():
                LoggingWrapper.error(line)
            sys.exit(testcommand_res.returncode)
        LoggingWrapper.info("The sequences are classified.", color="green", bold=True)
        for line in testcommand_res.stdout.splitlines():
            LoggingWrapper.info(line)
        testseqids = self.GetSeqIDs(seqrecords)
        predlabels, probas, ranks = self.GetPredictedLabels(testseqids, rdpclassifiedfilename)
        # load ref class dict
        classdict = {}
        with open(classesfilename) as classesfile:
            classdict = json.load(classesfile)
        self.SavePrediction(classdict, testseqids, testlabels, predlabels, probas, rdp_output)
        LoggingWrapper.info("The result is saved in the file: " + rdp_output, color="green", bold=True)