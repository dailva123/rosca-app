const CACHE_NAME = "rosca-app-v2"; // mude a versão quando atualizar
const urlsToCache = [
  "/",
  "/static/index.html",
  "/static/roscas_exemplo.png",
  "/static/icon-192.png",
  "/static/icon-512.png",
  "/static/manifest.json"
];

// Instalação do Service Worker (pré-cache dos arquivos)
self.addEventListener("install", event => {
  console.log("Service Worker: Instalando...");
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log("Service Worker: Cache aberto");
      return cache.addAll(urlsToCache);
    })
  );
});

// Ativação e remoção de caches antigos
self.addEventListener("activate", event => {
  console.log("Service Worker: Ativando e limpando caches antigos...");
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cache => {
          if (cache !== CACHE_NAME) {
            console.log("Service Worker: Deletando cache antigo:", cache);
            return caches.delete(cache);
          }
        })
      );
    })
  );
});

// Estratégia de cache: Offline First
self.addEventListener("fetch", event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      if (response) {
        // retorna do cache
        return response;
      }
      // se não estiver no cache, busca na rede e salva
      return fetch(event.request).then(fetchResponse => {
        return caches.open(CACHE_NAME).then(cache => {
          // Apenas salva arquivos GET no cache
          if (event.request.method === "GET") {
            cache.put(event.request, fetchResponse.clone());
          }
          return fetchResponse;
        });
      }).catch(() => {
        // fallback se offline e recurso não estiver no cache
        if (event.request.destination === "document") {
          return caches.match("/static/index.html");
        }
        if (event.request.destination === "image") {
          return caches.match("/static/icon-192.png"); // fallback imagem
        }
      });
    })
  );
});
