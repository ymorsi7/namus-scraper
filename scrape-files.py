import os, re, json, grequests, functools, mimetypes
import argparse

REQUEST_BATCH_SIZE = 10
REQUEST_FEEDBACK_INTERVAL = 50

USER_AGENT = "NamUs Scraper / github.com/prepager/namus-scraper"
DOWNLOAD_BASE_ENDPOINT = "https://www.namus.gov"

DATA_INPUT = "./output/{type}/{type}.json"
FILES_OUTPUT = "./output/{type}/files/{category}/{case}-{file}{extension}"
CASE_TYPES = ["MissingPersons", "UnidentifiedPersons", "UnclaimedPersons"]

KNOWN_EXTENSIONS = {"image/jpg": ".jpg", "image/pjpeg": ".jpg", "image/x-png": ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description='Download files for NamUs cases')
    parser.add_argument('--limit', type=int, help='Limit the number of cases to process')
    return parser.parse_args()


def main():
    args = parse_args()
    
    for caseType in CASE_TYPES:
        print("Collecting: {type}".format(type=caseType))

        cases = json.loads(open(DATA_INPUT.format(type=caseType), "r").read())
        
        if args.limit:
            cases = cases[:args.limit]
            print(f" > Limited to {args.limit} cases")
            
        print(" > Found %d cases" % len(cases))

        files = functools.reduce(
            lambda output, case: output
            + list(
                map(
                    lambda fileData: buildFile(caseType, case, fileData),
                    case["images"] + case["documents"],
                )
            ),
            cases,
            [],
        )
        print(" > Found %d files" % len(files))

        originalFileCount = len(files)
        files = list(
            filter(lambda fileData: not os.path.exists(fileData["path"]), files)
        )
        print(
            " > Skipped %d (%d) existing files"
            % (originalFileCount - len(files), len(files))
        )

        print(" > Starting file downloading")
        fileRequests = (
            grequests.get(fileData["url"], headers={"User-Agent": USER_AGENT})
            for fileData in files
        )
        requestUrlToIndexMap = {
            fileData["url"]: index for index, fileData in enumerate(files)
        }

        downloadedFiles = 0
        for response in grequests.imap(fileRequests, size=REQUEST_BATCH_SIZE):
            index = requestUrlToIndexMap[response.url]
            fileData = files[index]

            downloadedFiles = downloadedFiles + 1
            if downloadedFiles % REQUEST_FEEDBACK_INTERVAL == 0:
                print(" > Downloaded {count} files".format(count=downloadedFiles))

            if not response:
                print(
                    " > Failed downloading file: {file} index {index}".format(
                        file=fileData, index=index
                    )
                )
                continue

            os.makedirs(os.path.dirname(fileData["path"]), exist_ok=True)
            with open(fileData["path"], "wb") as outputFile:
                outputFile.write(response.content)

    print("Scraping completed")


def buildFile(caseType, case, fileData):
    return {
        **fileData,
        **{
            "caseId": case["id"],
            "url": DOWNLOAD_BASE_ENDPOINT + fileData["hrefDownload"],
            "path": buildFilePath(caseType, case, fileData),
        },
    }


def buildFilePath(caseType, case, fileData):
    originalFile = (
        fileData["files"]["original"] if "files" in fileData else fileData["file"]
    )
    mimetype = originalFile["mimeType"]
    fileName = originalFile["fileName"] if "fileName" in originalFile else None

    extensionFromFile = (
        "." + fileName.split(".")[-1]
        if fileName and re.match(r".+\.[a-zA-Z0-9]{2,4}$", fileName)
        else None
    )
    extension = (
        extensionFromFile
        or KNOWN_EXTENSIONS.get(mimetype)
        or mimetypes.guess_extension(mimetype)
    )

    if not extension:
        print(
            " > Unknown mimetype {type} for download {file}".format(
                type=mimetype, file=originalFile
            )
        )

    return FILES_OUTPUT.format(
        type=caseType,
        case=case["id"],
        file=originalFile["storageKey"],
        category=fileData["category"]["name"],
        extension=extension or ".unknown",
    )


main()
