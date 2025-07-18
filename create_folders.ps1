# Cria múltiplas pastas de ancestralidade com mais detalhes

$folders = @(
    # Europa
    "european_north_germanic",
    "european_north_scandinavian",
    "european_west_celtic",
    "european_south_iberian",
    "european_south_italian",
    "european_east_slavic",
    "european_jewish_ashkenazi",
    "european_jewish_sephardic",

    # África
    "african_west_niger_congo",
    "african_east_nilotic",
    "african_central_bantu",
    "afro_caribbean",
    "afro_brazilian",

    # América Latina
    "mestizo_latin_america",
    "mulatto_north_america",
    "indigenous_south_america_andes",
    "indigenous_south_america_amazon",

    # Ásia
    "asian_east_chinese",
    "asian_east_japanese",
    "asian_southeast_filipino",
    "asian_south_indian",
    "asian_south_pakistani",

    # Oriente Médio & Norte da África
    "mena_arab",
    "mena_berber",
    "mena_levantine",

    # Oceania & outros
    "polynesian",
    "melanesian",
    "native_north_american"
)

foreach ($folder in $folders) {
    mkdir $folder -Force
}
