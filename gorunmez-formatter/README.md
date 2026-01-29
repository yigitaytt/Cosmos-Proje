SFT sonucu model yalnizca '### Question' ve '### Answer' formatini ezberlemis olacakti ve buna uymayan bir formatta soru soruldugunda patlayacakti. Bunu engellemek icin chat.py ekledim. 

Kullanicinin girdisinin basina otomatik olarak '### Question' ve de sonuna otomatik olarak '### Answer' ekleyip o sekilde okumaya basliyor. Sonrasinda cevap verirken de bu kisim kirpilip sade bir sekilde cevabi yazdiriyor. Ayrica sira bazli bir sohpet sagliyo 'You:' ve 'Bot:' olarak. 

Bunun muadilini diger formatlayicida goremedim eger halihazirda yapiliyorsa bu dosyayi silebiliriz.