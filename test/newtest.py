from icrawler.builtin import BingImageCrawler

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for category in CIFAR_CLASSES:
    print(f"🔍 Crawling images for category: {category}")
    crawler = BingImageCrawler(storage={'root_dir': "."})
    crawler.crawl(keyword=category, max_num=5)
