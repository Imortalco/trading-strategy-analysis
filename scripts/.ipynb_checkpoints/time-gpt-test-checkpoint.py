from nixtla import NixtlaClient

nixtla_client = NixtlaClient(
    api_key= 'nixak-pMea9y5QDT7k02XdnzkgyI275sGJdg0IlaXoJ15UWRaPoKU3B07f6mDzpsb3Hf9ouLuMbJxpRg0nrd3g'
)

print(nixtla_client.validate_api_key(True))