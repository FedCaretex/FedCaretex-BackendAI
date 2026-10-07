// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title AuditTrail
/// @notice Lapisan integritas untuk federated learning: HANYA menyimpan hash
///         SHA-256 + metadata berukuran tetap. Tidak ada data pasien dan tidak
///         ada bobot model on-chain.
/// @dev Hanya `gateway` (server FL / gateway yang sudah memverifikasi tanda
///      tangan klien off-chain) yang boleh menulis. Timestamp diambil dari
///      block.timestamp, bukan dari parameter, agar tidak bisa dipalsukan.
contract AuditTrail {
    struct UpdateRecord {
        bytes32 hash;
        uint32 modelVersion;
        uint64 timestamp;
        bool exists;
    }

    struct GlobalRecord {
        bytes32 hash;
        uint32 modelVersion;
        uint64 timestamp;
        bool exists;
    }

    address public immutable gateway;

    // roundId => clientId (pseudonim, sha256 public key) => record
    mapping(uint256 => mapping(bytes32 => UpdateRecord)) private updates;
    mapping(uint256 => bytes32[]) private roundClients;
    mapping(uint256 => GlobalRecord) private globals;

    event ModelUpdateSubmitted(
        uint256 indexed roundId,
        bytes32 indexed clientId,
        uint32 modelVersion,
        bytes32 hash,
        uint64 timestamp
    );
    event GlobalModelSubmitted(
        uint256 indexed roundId,
        uint32 modelVersion,
        bytes32 hash,
        uint64 timestamp
    );

    error NotGateway();
    error AlreadyRecorded();
    error ZeroHash();

    modifier onlyGateway() {
        if (msg.sender != gateway) revert NotGateway();
        _;
    }

    constructor(address _gateway) {
        gateway = _gateway;
    }

    /// @notice Catat hash update model satu klien pada satu ronde.
    function submitModelUpdate(
        uint256 roundId,
        bytes32 clientId,
        uint32 modelVersion,
        bytes32 hash
    ) external onlyGateway {
        if (hash == bytes32(0)) revert ZeroHash();
        if (updates[roundId][clientId].exists) revert AlreadyRecorded();
        uint64 ts = uint64(block.timestamp);
        updates[roundId][clientId] = UpdateRecord(hash, modelVersion, ts, true);
        roundClients[roundId].push(clientId);
        emit ModelUpdateSubmitted(roundId, clientId, modelVersion, hash, ts);
    }

    /// @notice Catat hash model global hasil agregasi suatu ronde.
    function submitGlobalModel(
        uint256 roundId,
        uint32 modelVersion,
        bytes32 hash
    ) external onlyGateway {
        if (hash == bytes32(0)) revert ZeroHash();
        if (globals[roundId].exists) revert AlreadyRecorded();
        uint64 ts = uint64(block.timestamp);
        globals[roundId] = GlobalRecord(hash, modelVersion, ts, true);
        emit GlobalModelSubmitted(roundId, modelVersion, hash, ts);
    }

    /// @notice true jika hash yang diklaim sama dengan hash tercatat.
    function verifyIntegrity(
        uint256 roundId,
        bytes32 clientId,
        bytes32 hash
    ) external view returns (bool) {
        UpdateRecord storage r = updates[roundId][clientId];
        return r.exists && r.hash == hash;
    }

    /// @notice true jika hash model global yang diklaim sama dengan yang tercatat.
    function verifyGlobal(uint256 roundId, bytes32 hash) external view returns (bool) {
        GlobalRecord storage g = globals[roundId];
        return g.exists && g.hash == hash;
    }

    /// @notice Riwayat lengkap satu ronde (untuk audit pasca-hoc).
    /// @dev Ukuran array = jumlah klien per ronde (kecil pada skala eksperimen).
    function getAuditTrail(uint256 roundId)
        external
        view
        returns (
            bytes32[] memory clientIds,
            bytes32[] memory hashes,
            uint32[] memory modelVersions,
            uint64[] memory timestamps
        )
    {
        bytes32[] storage ids = roundClients[roundId];
        uint256 n = ids.length;
        clientIds = new bytes32[](n);
        hashes = new bytes32[](n);
        modelVersions = new uint32[](n);
        timestamps = new uint64[](n);
        for (uint256 i = 0; i < n; i++) {
            UpdateRecord storage r = updates[roundId][ids[i]];
            clientIds[i] = ids[i];
            hashes[i] = r.hash;
            modelVersions[i] = r.modelVersion;
            timestamps[i] = r.timestamp;
        }
    }

    function getGlobalModel(uint256 roundId)
        external
        view
        returns (bytes32 hash, uint32 modelVersion, uint64 timestamp, bool exists)
    {
        GlobalRecord storage g = globals[roundId];
        return (g.hash, g.modelVersion, g.timestamp, g.exists);
    }
}
