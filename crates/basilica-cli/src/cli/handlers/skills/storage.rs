//! Ownership-aware skill installation. Unknown or edited files are never removed.

use super::{join_rel, SkillFiles};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const RECEIPT: &str = ".basilica-install.json";

type Fingerprints = BTreeMap<PathBuf, String>;

#[derive(Deserialize)]
pub(super) struct Bundle {
    pub source: String,
    pub revision: String,
    pub skills: Vec<String>,
    pub files: BTreeMap<String, Fingerprints>,
    pub legacy_files: BTreeMap<String, Fingerprints>,
    pub legacy_playbook_sha256: String,
}

pub(super) fn bundle() -> Bundle {
    serde_json::from_str(include_str!("bundle.json")).expect("validated embedded skill manifest")
}

#[derive(Deserialize, Serialize)]
struct Receipt {
    source: String,
    revision: String,
    files: Fingerprints,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn fingerprints(files: &[(PathBuf, Vec<u8>)]) -> Fingerprints {
    files
        .iter()
        .map(|(path, bytes)| (path.clone(), digest(bytes)))
        .collect()
}

pub(super) fn verify_bundle(skills: &SkillFiles) -> io::Result<()> {
    let actual = skills
        .iter()
        .map(|(name, files)| (name.clone(), fingerprints(files)))
        .collect::<BTreeMap<_, _>>();
    if actual != bundle().files {
        return Err(io::Error::other(
            "Downloaded skills do not match the pinned bundle manifest",
        ));
    }
    Ok(())
}

// Reject symlinks (including the skill directory) rather than following them.
// Extra files also change ownership: an agent/user may have extended the skill.
fn tree_files(root: &Path) -> io::Result<Fingerprints> {
    fn visit(root: &Path, dir: &Path, files: &mut Fingerprints) -> io::Result<()> {
        if !fs::symlink_metadata(dir)?.file_type().is_dir() {
            return Err(io::Error::other("Skill path is not a regular directory"));
        }
        let mut entries = fs::read_dir(dir)?.peekable();
        if entries.peek().is_none() {
            return Err(io::Error::other(
                "Skill contains an unowned empty directory",
            ));
        }
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            if kind.is_dir() {
                visit(root, &path, files)?;
            } else if kind.is_file() {
                let relative = path.strip_prefix(root).unwrap();
                if relative != Path::new(RECEIPT) {
                    files.insert(relative.to_owned(), digest(&fs::read(path)?));
                }
            } else {
                return Err(io::Error::other("Skill contains a symlink or special file"));
            }
        }
        Ok(())
    }
    let mut files = Fingerprints::new();
    visit(root, root, &mut files)?;
    Ok(files)
}

pub(super) fn owned(path: &Path, name: &str) -> bool {
    let Ok(actual) = tree_files(path) else {
        return false;
    };
    let manifest = bundle();
    let receipt_path = path.join(RECEIPT);
    if let Ok(metadata) = fs::symlink_metadata(&receipt_path) {
        if !metadata.file_type().is_file() {
            return false;
        }
        let Ok(bytes) = fs::read(receipt_path) else {
            return false;
        };
        let Ok(receipt) = serde_json::from_slice::<Receipt>(&bytes) else {
            return false;
        };
        return receipt.source == manifest.source && receipt.files == actual;
    }
    manifest.legacy_files.get(name) == Some(&actual)
}

pub(super) fn installed_version(path: &Path) -> String {
    if !fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_dir())
        || !fs::symlink_metadata(path.join(RECEIPT))
            .is_ok_and(|metadata| metadata.file_type().is_file())
    {
        return "unrecorded (legacy or unmanaged)".to_string();
    }
    fs::read(path.join(RECEIPT))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Receipt>(&bytes).ok())
        .map(|receipt| receipt.revision)
        .unwrap_or_else(|| "unrecorded (legacy or unmanaged)".to_string())
}

pub(super) fn check_replace(path: &Path, name: &str) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
        Ok(_) if owned(path, name) => Ok(()),
        Ok(_) => Err(io::Error::other(format!(
            "Preserved unrecognized or modified skill {}. Move it to a backup location before retrying installation.",
            path.display()
        ))),
    }
}

pub(super) fn install(path: &Path, name: &str, files: &[(PathBuf, Vec<u8>)]) -> io::Result<()> {
    check_replace(path, name)?;
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("Missing skill parent"))?;
    fs::create_dir_all(parent)?;
    let staging = tempfile::Builder::new()
        .prefix(".basilica-stage-")
        .tempdir_in(parent)?;
    let staged = staging.path().join(name);
    fs::create_dir(&staged)?;
    for (rel, contents) in files {
        let destination = join_rel(&staged, rel);
        fs::create_dir_all(destination.parent().unwrap())?;
        fs::write(destination, contents)?;
    }
    let manifest = bundle();
    fs::write(
        staged.join(RECEIPT),
        serde_json::to_vec_pretty(&Receipt {
            source: manifest.source,
            revision: manifest.revision,
            files: fingerprints(files),
        })?,
    )?;
    let backup = staging.path().join("previous");
    let replacing = path.exists();
    if replacing {
        // Recheck after staging; do not overwrite edits made during download/write.
        check_replace(path, name)?;
        fs::rename(path, &backup)?;
    }
    if let Err(error) = fs::rename(&staged, path) {
        if replacing {
            if let Err(restore_error) = fs::rename(&backup, path) {
                let retained = staging.keep();
                return Err(io::Error::other(format!(
                    "Install failed: {error}; restore failed: {restore_error}. Previous skill retained at {}",
                    retained.join("previous").display()
                )));
            }
        }
        return Err(error);
    }
    Ok(())
}

pub(super) fn remove_owned(path: &Path, name: &str) -> io::Result<bool> {
    if owned(path, name) {
        fs::remove_dir_all(path)?;
        return Ok(true);
    }
    if fs::symlink_metadata(path).is_ok() {
        println!(
            "Preserved unrecognized or modified skill: {}",
            path.display()
        );
    }
    Ok(false)
}

pub(super) fn remove_legacy_playbook(root: &Path) -> io::Result<()> {
    let path = root.join("BASILICA-CLOUD-OPS.md");
    if let Ok(metadata) = fs::symlink_metadata(&path) {
        if metadata.file_type().is_file()
            && digest(&fs::read(&path)?) == bundle().legacy_playbook_sha256
        {
            fs::remove_file(path)?;
        } else {
            println!(
                "Preserved unrecognized or modified playbook: {}",
                path.display()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_upgrade_uninstall_preserves_modified_and_unrelated_content() {
        let root = tempfile::tempdir().unwrap();
        let skill = root.path().join("use-basilica");
        let unrelated = root.path().join("other");
        fs::create_dir(&unrelated).unwrap();
        fs::write(unrelated.join("SKILL.md"), "other").unwrap();
        install(
            &skill,
            "use-basilica",
            &[
                ("SKILL.md".into(), b"v1".to_vec()),
                ("old.md".into(), b"old".to_vec()),
            ],
        )
        .unwrap();
        assert!(owned(&skill, "use-basilica"));
        install(
            &skill,
            "use-basilica",
            &[("SKILL.md".into(), b"v2".to_vec())],
        )
        .unwrap();
        assert_eq!(fs::read(skill.join("SKILL.md")).unwrap(), b"v2");
        assert!(!skill.join("old.md").exists());
        let receipt: Receipt =
            serde_json::from_slice(&fs::read(skill.join(RECEIPT)).unwrap()).unwrap();
        assert_eq!(receipt.revision, bundle().revision);
        assert_eq!(receipt.source, bundle().source);
        fs::create_dir(skill.join("user-folder")).unwrap();
        assert!(!remove_owned(&skill, "use-basilica").unwrap());
        assert!(install(
            &skill,
            "use-basilica",
            &[("SKILL.md".into(), b"v3".to_vec())]
        )
        .is_err());
        assert!(skill.join("user-folder").is_dir());
        fs::remove_dir(skill.join("user-folder")).unwrap();
        fs::write(skill.join("notes.md"), "user notes").unwrap();
        assert!(!remove_owned(&skill, "use-basilica").unwrap());
        assert!(install(
            &skill,
            "use-basilica",
            &[("SKILL.md".into(), b"v3".to_vec())]
        )
        .is_err());
        assert_eq!(fs::read(skill.join("SKILL.md")).unwrap(), b"v2");
        fs::remove_file(skill.join("notes.md")).unwrap();
        fs::write(skill.join("SKILL.md"), "user edit").unwrap();
        assert!(!remove_owned(&skill, "use-basilica").unwrap());
        fs::write(skill.join("SKILL.md"), "v2").unwrap();
        assert!(remove_owned(&skill, "use-basilica").unwrap());
        assert_eq!(fs::read(unrelated.join("SKILL.md")).unwrap(), b"other");
    }

    #[test]
    fn legacy_snapshot_migrates_all_known_names_and_preserves_edits() {
        let home = tempfile::tempdir().unwrap();
        let root = home.path().join(".codex/skills");
        fs::create_dir_all(&root).unwrap();
        let decoder = flate2::read::GzDecoder::new(&include_bytes!("legacy-bundle.tar.gz")[..]);
        tar::Archive::new(decoder).unpack(&root).unwrap();
        let names = bundle()
            .legacy_files
            .keys()
            .filter(|name| name.as_str() != "use-basilica")
            .cloned()
            .collect::<Vec<_>>();
        for name in &names {
            assert!(
                owned(&root.join(name), name),
                "unrecognized legacy name {name}"
            );
        }
        let modified = root.join("basilica-sdk-ops/SKILL.md");
        fs::write(&modified, "user customization").unwrap();
        let tools = super::super::resolve_tools(home.path(), &["codex".to_string()]).unwrap();
        assert_eq!(
            super::super::clean_legacy(home.path(), &tools, false).unwrap(),
            4
        );
        assert_eq!(fs::read(&modified).unwrap(), b"user customization");
        assert!(!root.join("BASILICA-CLOUD-OPS.md").exists());
        for name in names {
            if name != "basilica-sdk-ops" {
                assert!(!root.join(name).exists());
            }
        }
    }

    #[test]
    fn malformed_receipts_and_unrecognized_skills_are_preserved() {
        let root = tempfile::tempdir().unwrap();
        let skill = root.path().join("use-basilica");
        fs::create_dir(&skill).unwrap();
        fs::write(skill.join("SKILL.md"), "custom").unwrap();
        assert!(check_replace(&skill, "use-basilica").is_err());
        fs::write(skill.join(RECEIPT), "invalid json").unwrap();
        assert!(!remove_owned(&skill, "use-basilica").unwrap());
        fs::write(root.path().join("BASILICA-CLOUD-OPS.md"), "personal notes").unwrap();
        remove_legacy_playbook(root.path()).unwrap();
        assert_eq!(
            fs::read(root.path().join("BASILICA-CLOUD-OPS.md")).unwrap(),
            b"personal notes"
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_skill_and_extra_symlink_are_preserved() {
        let root = tempfile::tempdir().unwrap();
        let skill = root.path().join("use-basilica");
        let outside = root.path().join("outside");
        install(
            &outside,
            "use-basilica",
            &[("SKILL.md".into(), b"content".to_vec())],
        )
        .unwrap();
        std::os::unix::fs::symlink(&outside, &skill).unwrap();
        assert!(check_replace(&skill, "use-basilica").is_err());
        assert!(!remove_owned(&skill, "use-basilica").unwrap());
        fs::remove_file(&skill).unwrap();
        install(
            &skill,
            "use-basilica",
            &[("SKILL.md".into(), b"content".to_vec())],
        )
        .unwrap();
        std::os::unix::fs::symlink(&outside, skill.join("notes")).unwrap();
        assert!(!remove_owned(&skill, "use-basilica").unwrap());
        assert!(outside.join("SKILL.md").exists());
    }
}
