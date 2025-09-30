//! Country name to ISO 3166-1 alpha-2 code mapping

use std::collections::HashMap;

use once_cell::sync::Lazy;

/// Single source of truth for country data
/// Format: (ISO code, Primary name, Aliases)
const COUNTRIES: &[(&str, &str, &[&str])] = &[
    ("AF", "Afghanistan", &["afghanistan"]),
    ("AL", "Albania", &["albania"]),
    ("DZ", "Algeria", &["algeria"]),
    ("AD", "Andorra", &["andorra"]),
    ("AO", "Angola", &["angola"]),
    ("AG", "Antigua and Barbuda", &["antigua and barbuda"]),
    ("AR", "Argentina", &["argentina"]),
    ("AM", "Armenia", &["armenia"]),
    ("AU", "Australia", &["australia"]),
    ("AT", "Austria", &["austria"]),
    ("AZ", "Azerbaijan", &["azerbaijan"]),
    ("BS", "Bahamas", &["bahamas"]),
    ("BH", "Bahrain", &["bahrain"]),
    ("BD", "Bangladesh", &["bangladesh"]),
    ("BB", "Barbados", &["barbados"]),
    ("BY", "Belarus", &["belarus"]),
    ("BE", "Belgium", &["belgium"]),
    ("BZ", "Belize", &["belize"]),
    ("BJ", "Benin", &["benin"]),
    ("BT", "Bhutan", &["bhutan"]),
    ("BO", "Bolivia", &["bolivia"]),
    ("BA", "Bosnia and Herzegovina", &["bosnia and herzegovina"]),
    ("BW", "Botswana", &["botswana"]),
    ("BR", "Brazil", &["brazil"]),
    ("BN", "Brunei", &["brunei"]),
    ("BG", "Bulgaria", &["bulgaria"]),
    ("BF", "Burkina Faso", &["burkina faso"]),
    ("BI", "Burundi", &["burundi"]),
    ("KH", "Cambodia", &["cambodia"]),
    ("CM", "Cameroon", &["cameroon"]),
    ("CA", "Canada", &["canada"]),
    ("CV", "Cape Verde", &["cape verde"]),
    (
        "CF",
        "Central African Republic",
        &["central african republic"],
    ),
    ("TD", "Chad", &["chad"]),
    ("CL", "Chile", &["chile"]),
    ("CN", "China", &["china"]),
    ("CO", "Colombia", &["colombia"]),
    ("KM", "Comoros", &["comoros"]),
    ("CG", "Congo", &["congo"]),
    (
        "CD",
        "Democratic Republic of the Congo",
        &[
            "democratic republic of the congo",
            "congo, democratic republic",
        ],
    ),
    ("CR", "Costa Rica", &["costa rica"]),
    ("HR", "Croatia", &["croatia"]),
    ("CU", "Cuba", &["cuba"]),
    ("CY", "Cyprus", &["cyprus"]),
    ("CZ", "Czech Republic", &["czech republic", "czechia"]),
    ("DK", "Denmark", &["denmark"]),
    ("DJ", "Djibouti", &["djibouti"]),
    ("DM", "Dominica", &["dominica"]),
    ("DO", "Dominican Republic", &["dominican republic"]),
    ("TL", "East Timor", &["east timor", "timor-leste"]),
    ("EC", "Ecuador", &["ecuador"]),
    ("EG", "Egypt", &["egypt"]),
    ("SV", "El Salvador", &["el salvador"]),
    ("GQ", "Equatorial Guinea", &["equatorial guinea"]),
    ("ER", "Eritrea", &["eritrea"]),
    ("EE", "Estonia", &["estonia"]),
    ("ET", "Ethiopia", &["ethiopia"]),
    ("FJ", "Fiji", &["fiji"]),
    ("FI", "Finland", &["finland"]),
    ("FR", "France", &["france"]),
    ("GA", "Gabon", &["gabon"]),
    ("GM", "Gambia", &["gambia"]),
    ("GE", "Georgia", &["georgia"]),
    ("DE", "Germany", &["germany"]),
    ("GH", "Ghana", &["ghana"]),
    ("GR", "Greece", &["greece"]),
    ("GD", "Grenada", &["grenada"]),
    ("GT", "Guatemala", &["guatemala"]),
    ("GN", "Guinea", &["guinea"]),
    ("GW", "Guinea-Bissau", &["guinea-bissau"]),
    ("GY", "Guyana", &["guyana"]),
    ("HT", "Haiti", &["haiti"]),
    ("HN", "Honduras", &["honduras"]),
    ("HU", "Hungary", &["hungary"]),
    ("IS", "Iceland", &["iceland"]),
    ("IN", "India", &["india"]),
    ("ID", "Indonesia", &["indonesia"]),
    ("IR", "Iran", &["iran"]),
    ("IQ", "Iraq", &["iraq"]),
    ("IE", "Ireland", &["ireland"]),
    ("IL", "Israel", &["israel"]),
    ("IT", "Italy", &["italy"]),
    ("CI", "Ivory Coast", &["ivory coast"]),
    ("JM", "Jamaica", &["jamaica"]),
    ("JP", "Japan", &["japan"]),
    ("JO", "Jordan", &["jordan"]),
    ("KZ", "Kazakhstan", &["kazakhstan"]),
    ("KE", "Kenya", &["kenya"]),
    ("KI", "Kiribati", &["kiribati"]),
    ("KP", "North Korea", &["north korea", "korea, north"]),
    ("KR", "South Korea", &["south korea", "korea, south"]),
    ("XK", "Kosovo", &["kosovo"]),
    ("KW", "Kuwait", &["kuwait"]),
    ("KG", "Kyrgyzstan", &["kyrgyzstan"]),
    ("LA", "Laos", &["laos"]),
    ("LV", "Latvia", &["latvia"]),
    ("LB", "Lebanon", &["lebanon"]),
    ("LS", "Lesotho", &["lesotho"]),
    ("LR", "Liberia", &["liberia"]),
    ("LY", "Libya", &["libya"]),
    ("LI", "Liechtenstein", &["liechtenstein"]),
    ("LT", "Lithuania", &["lithuania"]),
    ("LU", "Luxembourg", &["luxembourg"]),
    ("MK", "North Macedonia", &["north macedonia", "macedonia"]),
    ("MG", "Madagascar", &["madagascar"]),
    ("MW", "Malawi", &["malawi"]),
    ("MY", "Malaysia", &["malaysia"]),
    ("MV", "Maldives", &["maldives"]),
    ("ML", "Mali", &["mali"]),
    ("MT", "Malta", &["malta"]),
    ("MH", "Marshall Islands", &["marshall islands"]),
    ("MR", "Mauritania", &["mauritania"]),
    ("MU", "Mauritius", &["mauritius"]),
    ("MX", "Mexico", &["mexico"]),
    ("FM", "Micronesia", &["micronesia"]),
    ("MD", "Moldova", &["moldova"]),
    ("MC", "Monaco", &["monaco"]),
    ("MN", "Mongolia", &["mongolia"]),
    ("ME", "Montenegro", &["montenegro"]),
    ("MA", "Morocco", &["morocco"]),
    ("MZ", "Mozambique", &["mozambique"]),
    ("MM", "Myanmar", &["myanmar", "burma"]),
    ("NA", "Namibia", &["namibia"]),
    ("NR", "Nauru", &["nauru"]),
    ("NP", "Nepal", &["nepal"]),
    ("NL", "Netherlands", &["netherlands", "holland"]),
    ("NZ", "New Zealand", &["new zealand"]),
    ("NI", "Nicaragua", &["nicaragua"]),
    ("NE", "Niger", &["niger"]),
    ("NG", "Nigeria", &["nigeria"]),
    ("NO", "Norway", &["norway"]),
    ("OM", "Oman", &["oman"]),
    ("PK", "Pakistan", &["pakistan"]),
    ("PW", "Palau", &["palau"]),
    ("PS", "Palestine", &["palestine"]),
    ("PA", "Panama", &["panama"]),
    ("PG", "Papua New Guinea", &["papua new guinea"]),
    ("PY", "Paraguay", &["paraguay"]),
    ("PE", "Peru", &["peru"]),
    ("PH", "Philippines", &["philippines"]),
    ("PL", "Poland", &["poland"]),
    ("PT", "Portugal", &["portugal"]),
    ("QA", "Qatar", &["qatar"]),
    ("RO", "Romania", &["romania"]),
    ("RU", "Russia", &["russia"]),
    ("RW", "Rwanda", &["rwanda"]),
    ("KN", "Saint Kitts and Nevis", &["saint kitts and nevis"]),
    ("LC", "Saint Lucia", &["saint lucia"]),
    (
        "VC",
        "Saint Vincent and the Grenadines",
        &["saint vincent and the grenadines"],
    ),
    ("WS", "Samoa", &["samoa"]),
    ("SM", "San Marino", &["san marino"]),
    ("ST", "Sao Tome and Principe", &["sao tome and principe"]),
    ("SA", "Saudi Arabia", &["saudi arabia"]),
    ("SN", "Senegal", &["senegal"]),
    ("RS", "Serbia", &["serbia"]),
    ("SC", "Seychelles", &["seychelles"]),
    ("SL", "Sierra Leone", &["sierra leone"]),
    ("SG", "Singapore", &["singapore"]),
    ("SK", "Slovakia", &["slovakia"]),
    ("SI", "Slovenia", &["slovenia"]),
    ("SB", "Solomon Islands", &["solomon islands"]),
    ("SO", "Somalia", &["somalia"]),
    ("ZA", "South Africa", &["south africa"]),
    ("SS", "South Sudan", &["south sudan"]),
    ("ES", "Spain", &["spain"]),
    ("LK", "Sri Lanka", &["sri lanka"]),
    ("SD", "Sudan", &["sudan"]),
    ("SR", "Suriname", &["suriname"]),
    ("SZ", "Eswatini", &["eswatini", "swaziland"]),
    ("SE", "Sweden", &["sweden"]),
    ("CH", "Switzerland", &["switzerland"]),
    ("SY", "Syria", &["syria"]),
    ("TW", "Taiwan", &["taiwan"]),
    ("TJ", "Tajikistan", &["tajikistan"]),
    ("TZ", "Tanzania", &["tanzania"]),
    ("TH", "Thailand", &["thailand"]),
    ("TG", "Togo", &["togo"]),
    ("TO", "Tonga", &["tonga"]),
    ("TT", "Trinidad and Tobago", &["trinidad and tobago"]),
    ("TN", "Tunisia", &["tunisia"]),
    ("TR", "Turkey", &["turkey"]),
    ("TM", "Turkmenistan", &["turkmenistan"]),
    ("TV", "Tuvalu", &["tuvalu"]),
    ("UG", "Uganda", &["uganda"]),
    ("UA", "Ukraine", &["ukraine"]),
    (
        "AE",
        "United Arab Emirates",
        &["united arab emirates", "uae"],
    ),
    (
        "GB",
        "United Kingdom",
        &["united kingdom", "uk", "great britain", "england"],
    ),
    (
        "US",
        "United States",
        &[
            "united states",
            "usa",
            "united states of america",
            "america",
        ],
    ),
    ("UY", "Uruguay", &["uruguay"]),
    ("UZ", "Uzbekistan", &["uzbekistan"]),
    ("VU", "Vanuatu", &["vanuatu"]),
    ("VA", "Vatican City", &["vatican city", "vatican"]),
    ("VE", "Venezuela", &["venezuela"]),
    ("VN", "Vietnam", &["vietnam"]),
    ("YE", "Yemen", &["yemen"]),
    ("ZM", "Zambia", &["zambia"]),
    ("ZW", "Zimbabwe", &["zimbabwe"]),
];

/// Convert a country name or code to its ISO 3166-1 alpha-2 code
/// Returns the input unchanged if no mapping is found (assumes it's already a code)
pub fn normalize_country_code(input: &str) -> String {
    let input_lower = input.to_lowercase();

    // Try to find in our mapping first (handles special cases like "UK" -> "GB")
    if let Some(code) = COUNTRY_MAPPINGS.get(input_lower.as_str()) {
        return code.to_string();
    }

    // If it's a 2-letter code not in our mapping, just return uppercase version
    if input.len() == 2 {
        return input.to_uppercase();
    }

    // Return the input uppercased if not found
    input.to_uppercase()
}

/// Convert an ISO 3166-1 alpha-2 code to its full country name
/// Returns the input unchanged if no mapping is found
pub fn get_country_name_from_code(code: &str) -> String {
    let code_upper = code.to_uppercase();

    // Try to find in our reverse mapping
    if let Some(name) = CODE_TO_COUNTRY.get(code_upper.as_str()) {
        return name.to_string();
    }

    // Return the input if not found
    code.to_string()
}

/// Mapping from country names/aliases to ISO codes
static COUNTRY_MAPPINGS: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();

    // Build from the COUNTRIES const array
    for (code, _primary_name, aliases) in COUNTRIES {
        for alias in *aliases {
            m.insert(*alias, *code);
        }
    }

    m
});

/// Mapping from ISO codes to primary country names
static CODE_TO_COUNTRY: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();

    // Build from the COUNTRIES const array
    for (code, primary_name, _aliases) in COUNTRIES {
        m.insert(*code, *primary_name);
    }

    m
});
